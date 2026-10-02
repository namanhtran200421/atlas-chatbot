import { TestBed } from '@angular/core/testing';

import { environment } from '../../environments/environment';
import { AuthService, AuthenticationError } from './auth.service';

describe('AuthService', () => {
  const originalClientId = environment.cognitoClientId;
  const originalRegion = environment.cognitoRegion;

  beforeEach(() => {
    sessionStorage.clear();
    environment.cognitoClientId = 'browser-client-id';
    environment.cognitoRegion = 'ap-southeast-2';
    TestBed.configureTestingModule({});
  });

  afterEach(() => {
    TestBed.inject(AuthService).signOut();
    environment.cognitoClientId = originalClientId;
    environment.cognitoRegion = originalRegion;
    vi.unstubAllGlobals();
  });

  it('signs in with email and password and keeps the access token in session storage', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          AuthenticationResult: {
            AccessToken: 'access-token',
            IdToken: 'id-token',
            ExpiresIn: 3600,
          },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const auth = TestBed.inject(AuthService);
    await auth.signIn('member@example.com', 'correct horse battery staple');

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://cognito-idp.ap-southeast-2.amazonaws.com/');
    expect(JSON.parse(String(options.body))).toEqual({
      AuthFlow: 'USER_PASSWORD_AUTH',
      ClientId: 'browser-client-id',
      AuthParameters: {
        USERNAME: 'member@example.com',
        PASSWORD: 'correct horse battery staple',
      },
    });
    expect(auth.accessToken()).toBe('access-token');
    expect(sessionStorage.getItem('oriana-cognito-session')).toContain('access-token');
  });

  it('does not reveal whether a user exists when Cognito rejects the credentials', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ __type: 'UserNotFoundException' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    const auth = TestBed.inject(AuthService);
    await expect(auth.signIn('missing@example.com', 'wrong')).rejects.toEqual(
      new AuthenticationError('The email or password is incorrect.'),
    );
  });

  it('explains when password authentication is disabled for the app client', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            __type: 'InvalidParameterException',
            message: 'USER_PASSWORD_AUTH flow not enabled for this client',
          }),
          { status: 400, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );

    const auth = TestBed.inject(AuthService);
    await expect(auth.signIn('member@example.com', 'password')).rejects.toEqual(
      new AuthenticationError(
        'Password sign-in is not enabled for this Cognito app client. Enable ALLOW_USER_PASSWORD_AUTH in its authentication-flow settings.',
      ),
    );
  });

  it('signs up with an email attribute and confirms the emailed code', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            UserConfirmed: false,
            CodeDeliveryDetails: {
              AttributeName: 'email',
              DeliveryMedium: 'EMAIL',
              Destination: 'm***@e***',
            },
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      )
      .mockResolvedValueOnce(
        new Response('{}', {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const auth = TestBed.inject(AuthService);
    await expect(auth.signUp('member@example.com', 'StrongPassword1!')).resolves.toEqual({
      userConfirmed: false,
      destination: 'm***@e***',
    });
    await auth.confirmSignUp('member@example.com', '123456');

    const [, signUpOptions] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect((signUpOptions.headers as Record<string, string>)['X-Amz-Target']).toBe(
      'AWSCognitoIdentityProviderService.SignUp',
    );
    expect(JSON.parse(String(signUpOptions.body))).toEqual({
      ClientId: 'browser-client-id',
      Username: 'member@example.com',
      Password: 'StrongPassword1!',
      UserAttributes: [{ Name: 'email', Value: 'member@example.com' }],
    });

    const [, confirmOptions] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect((confirmOptions.headers as Record<string, string>)['X-Amz-Target']).toBe(
      'AWSCognitoIdentityProviderService.ConfirmSignUp',
    );
    expect(JSON.parse(String(confirmOptions.body))).toEqual({
      ClientId: 'browser-client-id',
      Username: 'member@example.com',
      ConfirmationCode: '123456',
    });
  });

  it('resends a confirmation code to an unconfirmed user', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ CodeDeliveryDetails: { Destination: 'm***@e***' } }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const auth = TestBed.inject(AuthService);
    await expect(auth.resendConfirmationCode('member@example.com')).resolves.toBe('m***@e***');

    const [, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect((options.headers as Record<string, string>)['X-Amz-Target']).toBe(
      'AWSCognitoIdentityProviderService.ResendConfirmationCode',
    );
  });

  it('clears the browser session on sign out', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({ AuthenticationResult: { AccessToken: 'token', ExpiresIn: 3600 } }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        ),
    );

    const auth = TestBed.inject(AuthService);
    await auth.signIn('member@example.com', 'password');
    auth.signOut();

    expect(auth.hasValidSession()).toBe(false);
    expect(sessionStorage.getItem('oriana-cognito-session')).toBeNull();
  });
});
