import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';

import { environment } from '../../environments/environment';
import { AuthService } from './auth.service';

/** Adds the Cognito access token to chat calls and handles rejected sessions. */
export const authInterceptor: HttpInterceptorFn = (request, next) => {
  const auth = inject(AuthService);
  const router = inject(Router);
  const isChatRequest = request.url === environment.chatApiUrl;
  const token = isChatRequest ? auth.accessToken() : null;
  const authenticatedRequest = token
    ? request.clone({ setHeaders: { Authorization: `Bearer ${token}` } })
    : request;

  return next(authenticatedRequest).pipe(
    catchError((error: unknown) => {
      if (isRejectedSession(error)) {
        auth.signOut();
        void router.navigate(['/login'], { queryParams: { reason: 'session-expired' } });
      }
      return throwError(() => error);
    }),
  );
};

function isRejectedSession(error: unknown): boolean {
  if (!(error instanceof HttpErrorResponse)) return false;

  const apiCode =
    typeof error.error === 'object' && error.error !== null && 'error' in error.error
      ? String((error.error as { error?: unknown }).error ?? '')
      : '';

  if (apiCode === 'credentials_unavailable' || apiCode === 'request_not_allowed') return false;
  return error.status === 401 || (error.status === 403 && !apiCode);
}
