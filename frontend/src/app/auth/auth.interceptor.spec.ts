import { HttpClient, provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { environment } from '../../environments/environment';
import { AuthService } from './auth.service';
import { authInterceptor } from './auth.interceptor';

describe('authInterceptor', () => {
  let http: HttpTestingController;
  const auth = {
    accessToken: vi.fn(() => 'access-token'),
    signOut: vi.fn(),
  };

  beforeEach(() => {
    auth.accessToken.mockClear();
    auth.signOut.mockClear();
    TestBed.configureTestingModule({
      providers: [
        provideRouter([]),
        { provide: AuthService, useValue: auth },
        provideHttpClient(withInterceptors([authInterceptor])),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('adds the Cognito access token to chat requests', () => {
    const client = TestBed.inject(HttpClient);

    client.post(environment.chatApiUrl, { query: 'Hello' }).subscribe();
    const request = http.expectOne(environment.chatApiUrl);

    expect(request.request.headers.get('Authorization')).toBe('Bearer access-token');
    request.flush({ answer: 'Hello' });
  });

  it('does not attach a token to public chat requests', () => {
    const client = TestBed.inject(HttpClient);
    client.post(environment.publicChatApiUrl, { query: 'Hello' }).subscribe();
    const request = http.expectOne(environment.publicChatApiUrl);
    expect(request.request.headers.has('Authorization')).toBe(false);
    request.flush({ answer: 'Hello' });
  });
});
