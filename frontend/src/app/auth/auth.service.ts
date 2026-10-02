import { Injectable, computed, signal } from '@angular/core';

import { environment } from '../../environments/environment';

const SESSION_STORAGE_KEY = 'oriana-cognito-session';
const EXPIRY_SKEW_MS = 15_000;

interface CognitoAuthenticationResult {
  AccessToken?: string;
  IdToken?: string;
  RefreshToken?: string;
  ExpiresIn?: number;
}

interface CognitoAuthResponse {
  AuthenticationResult?: CognitoAuthenticationResult;
  ChallengeName?: string;
  UserConfirmed?: boolean;
  CodeDeliveryDetails?: {
    AttributeName?: string;
    DeliveryMedium?: string;
    Destination?: string;
  };
  __type?: string;
  message?: string;
}

export interface AuthSession {
  email: string;
  accessToken: string;
  idToken?: string;
  refreshToken?: string;
  expiresAt: number;
}

export class AuthenticationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AuthenticationError';
  }
}

export interface SignUpResult {
  userConfirmed: boolean;
  destination?: string;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly _session = signal<AuthSession | null>(this.loadSession());
  private expiryTimer: ReturnType<typeof setTimeout> | undefined;

  readonly session = this._session.asReadonly();
  readonly email = computed(() => this._session()?.email ?? '');
  readonly isAuthenticated = computed(() => this._session() !== null);

  constructor() {
    this.scheduleExpiry(this._session());
  }

  hasValidSession(): boolean {
    return this.validSession() !== null;
  }

  accessToken(): string | null {
    return this.validSession()?.accessToken ?? null;
  }

  async signIn(email: string, password: string): Promise<void> {
    const clientId = environment.cognitoClientId.trim();
    const data = await this.cognitoRequest('InitiateAuth', {
      AuthFlow: 'USER_PASSWORD_AUTH',
      ClientId: clientId,
      AuthParameters: {
        USERNAME: email,
        PASSWORD: password,
      },
    });

    const result = data.AuthenticationResult;
    if (!result?.AccessToken) {
      const challengeName = data.ChallengeName ? ` (${data.ChallengeName})` : '';
      throw new AuthenticationError(
        `Your account requires another sign-in step${challengeName}. Ask an administrator to complete it before using Oriana.`,
      );
    }

    this.saveSession(email, result);
  }

  async signUp(email: string, password: string): Promise<SignUpResult> {
    const clientId = environment.cognitoClientId.trim();
    const data = await this.cognitoRequest('SignUp', {
      ClientId: clientId,
      Username: email,
      Password: password,
      UserAttributes: [{ Name: 'email', Value: email }],
    });

    return {
      userConfirmed: data.UserConfirmed === true,
      ...(data.CodeDeliveryDetails?.Destination
        ? { destination: data.CodeDeliveryDetails.Destination }
        : {}),
    };
  }

  async confirmSignUp(email: string, confirmationCode: string): Promise<void> {
    const clientId = environment.cognitoClientId.trim();
    await this.cognitoRequest('ConfirmSignUp', {
      ClientId: clientId,
      Username: email,
      ConfirmationCode: confirmationCode,
    });
  }

  async resendConfirmationCode(email: string): Promise<string | undefined> {
    const clientId = environment.cognitoClientId.trim();
    const data = await this.cognitoRequest('ResendConfirmationCode', {
      ClientId: clientId,
      Username: email,
    });
    return data.CodeDeliveryDetails?.Destination;
  }

  private saveSession(email: string, result: CognitoAuthenticationResult): void {
    if (!result.AccessToken) return;

    const session: AuthSession = {
      email,
      accessToken: result.AccessToken,
      ...(result.IdToken ? { idToken: result.IdToken } : {}),
      ...(result.RefreshToken ? { refreshToken: result.RefreshToken } : {}),
      expiresAt: Date.now() + Number(result.ExpiresIn ?? 3600) * 1000,
    };

    this.storeSession(session);
    this._session.set(session);
    this.scheduleExpiry(session);
  }

  private async cognitoRequest(
    target: 'InitiateAuth' | 'SignUp' | 'ConfirmSignUp' | 'ResendConfirmationCode',
    body: Record<string, unknown>,
  ): Promise<CognitoAuthResponse> {
    const clientId = environment.cognitoClientId.trim();
    const region = environment.cognitoRegion.trim();

    if (!clientId || !region) {
      throw new AuthenticationError(
        'Authentication is not configured. Set cognitoClientId and cognitoRegion in the Angular environment file.',
      );
    }

    let response: Response;
    try {
      response = await fetch(`https://cognito-idp.${region}.amazonaws.com/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-amz-json-1.1',
          'X-Amz-Target': `AWSCognitoIdentityProviderService.${target}`,
        },
        body: JSON.stringify(body),
      });
    } catch {
      throw new AuthenticationError(
        'Cognito could not be reached. Check your connection and AWS region, then try again.',
      );
    }

    const data = (await response.json().catch(() => ({}))) as CognitoAuthResponse;
    if (!response.ok) {
      throw new AuthenticationError(this.cognitoErrorMessage(data, target));
    }

    return data;
  }

  signOut(): void {
    if (this.expiryTimer) clearTimeout(this.expiryTimer);
    this.expiryTimer = undefined;
    this.removeStoredSession();
    this._session.set(null);
  }

  private validSession(): AuthSession | null {
    const session = this._session();
    if (!session || session.expiresAt <= Date.now() + EXPIRY_SKEW_MS) {
      if (session) this.signOut();
      return null;
    }
    return session;
  }

  private loadSession(): AuthSession | null {
    try {
      const raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
      if (!raw) return null;

      const session = JSON.parse(raw) as AuthSession;
      if (
        !session.email ||
        !session.accessToken ||
        !Number.isFinite(session.expiresAt) ||
        session.expiresAt <= Date.now() + EXPIRY_SKEW_MS
      ) {
        sessionStorage.removeItem(SESSION_STORAGE_KEY);
        return null;
      }
      return session;
    } catch {
      this.removeStoredSession();
      return null;
    }
  }

  private storeSession(session: AuthSession): void {
    try {
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
    } catch {
      // The in-memory session still works when browser storage is unavailable.
    }
  }

  private removeStoredSession(): void {
    try {
      sessionStorage.removeItem(SESSION_STORAGE_KEY);
    } catch {
      // There is nothing else to clear.
    }
  }

  private scheduleExpiry(session: AuthSession | null): void {
    if (this.expiryTimer) clearTimeout(this.expiryTimer);
    this.expiryTimer = undefined;
    if (!session) return;

    const delay = Math.max(0, session.expiresAt - Date.now() - EXPIRY_SKEW_MS);
    this.expiryTimer = setTimeout(() => this.signOut(), Math.min(delay, 2_147_483_647));
  }

  private cognitoErrorMessage(
    data: CognitoAuthResponse,
    target: 'InitiateAuth' | 'SignUp' | 'ConfirmSignUp' | 'ResendConfirmationCode',
  ): string {
    const type = String(data.__type ?? '')
      .split('#')
      .pop();
    if (type === 'NotAuthorizedException' && target === 'SignUp') {
      return 'Self-service sign-up is not enabled for this Cognito user pool.';
    }
    if (
      type === 'InvalidParameterException' &&
      target === 'InitiateAuth' &&
      /USER_PASSWORD_AUTH flow not enabled/i.test(data.message ?? '')
    ) {
      return 'Password sign-in is not enabled for this Cognito app client. Enable ALLOW_USER_PASSWORD_AUTH in its authentication-flow settings.';
    }

    const messages: Record<string, string> = {
      NotAuthorizedException: 'The email or password is incorrect.',
      UserNotConfirmedException: 'This account has not been confirmed yet.',
      PasswordResetRequiredException: 'You need to reset your password before signing in.',
      InvalidPasswordException: 'That password does not meet the account password requirements.',
      UsernameExistsException: 'An account with that email address already exists.',
      CodeMismatchException: 'That confirmation code is incorrect. Check the email and try again.',
      ExpiredCodeException: 'That confirmation code has expired. Request a new code and try again.',
      LimitExceededException: 'Too many attempts. Wait a moment and try again.',
      CodeDeliveryFailureException:
        'Cognito could not deliver the confirmation email. Try again later.',
      TooManyRequestsException: 'Too many attempts. Wait a moment and try again.',
      UserNotFoundException: 'The email or password is incorrect.',
      InvalidParameterException: 'Cognito could not process those sign-in details.',
    };

    return messages[type ?? ''] ?? data.message ?? 'Cognito could not complete the request.';
  }
}
