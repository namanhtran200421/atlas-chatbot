import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';

import { AuthService } from '../auth/auth.service';

@Component({
  selector: 'app-login',
  imports: [ReactiveFormsModule, RouterLink],
  templateUrl: './login.html',
  styleUrl: './login.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Login {
  private readonly auth = inject(AuthService);
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);

  protected readonly form = new FormGroup({
    email: new FormControl('', {
      nonNullable: true,
      validators: [Validators.required, Validators.email],
    }),
    password: new FormControl('', {
      nonNullable: true,
      validators: [Validators.required],
    }),
  });
  protected readonly isSubmitting = signal(false);
  protected readonly errorMessage = signal(
    this.route.snapshot.queryParamMap.get('reason') === 'session-expired'
      ? 'Your session has expired. Sign in again to continue.'
      : '',
  );
  protected readonly statusMessage = signal(
    this.route.snapshot.queryParamMap.get('signup') === 'confirmed'
      ? 'Your email is confirmed. You can sign in now.'
      : '',
  );

  protected async submit(): Promise<void> {
    if (this.form.invalid || this.isSubmitting()) {
      this.form.markAllAsTouched();
      return;
    }

    this.isSubmitting.set(true);
    this.errorMessage.set('');

    try {
      const { email, password } = this.form.getRawValue();
      await this.auth.signIn(email.trim(), password);
      this.form.controls.password.reset();

      const requestedUrl = this.route.snapshot.queryParamMap.get('returnUrl');
      const returnUrl =
        requestedUrl?.startsWith('/') && !requestedUrl.startsWith('//') ? requestedUrl : '/';
      await this.router.navigateByUrl(returnUrl);
    } catch (error) {
      this.errorMessage.set(
        error instanceof Error ? error.message : 'Cognito could not sign you in.',
      );
    } finally {
      this.isSubmitting.set(false);
    }
  }
}
