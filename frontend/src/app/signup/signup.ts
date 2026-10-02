import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';

import { AuthService } from '../auth/auth.service';

@Component({
  selector: 'app-signup',
  imports: [ReactiveFormsModule, RouterLink],
  templateUrl: './signup.html',
  styleUrl: '../login/login.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Signup {
  private readonly auth = inject(AuthService);
  private readonly router = inject(Router);

  protected readonly detailsForm = new FormGroup({
    email: new FormControl('', {
      nonNullable: true,
      validators: [Validators.required, Validators.email],
    }),
    password: new FormControl('', {
      nonNullable: true,
      validators: [Validators.required],
    }),
    confirmPassword: new FormControl('', {
      nonNullable: true,
      validators: [Validators.required],
    }),
  });
  protected readonly confirmationForm = new FormGroup({
    code: new FormControl('', {
      nonNullable: true,
      validators: [Validators.required],
    }),
  });

  protected readonly confirmationEmail = signal('');
  protected readonly deliveryDestination = signal('');
  protected readonly isSubmitting = signal(false);
  protected readonly isResending = signal(false);
  protected readonly errorMessage = signal('');
  protected readonly resendMessage = signal('');

  protected passwordsDiffer(): boolean {
    const { password, confirmPassword } = this.detailsForm.getRawValue();
    return Boolean(confirmPassword) && password !== confirmPassword;
  }

  protected async signUp(): Promise<void> {
    if (this.detailsForm.invalid || this.passwordsDiffer() || this.isSubmitting()) {
      this.detailsForm.markAllAsTouched();
      return;
    }

    this.isSubmitting.set(true);
    this.errorMessage.set('');

    try {
      const { email, password } = this.detailsForm.getRawValue();
      const normalizedEmail = email.trim();
      const result = await this.auth.signUp(normalizedEmail, password);

      if (result.userConfirmed) {
        await this.router.navigate(['/login'], { queryParams: { signup: 'confirmed' } });
        return;
      }

      this.confirmationEmail.set(normalizedEmail);
      this.deliveryDestination.set(result.destination ?? normalizedEmail);
      this.detailsForm.controls.password.reset();
      this.detailsForm.controls.confirmPassword.reset();
    } catch (error) {
      this.errorMessage.set(
        error instanceof Error ? error.message : 'Cognito could not create your account.',
      );
    } finally {
      this.isSubmitting.set(false);
    }
  }

  protected async confirm(): Promise<void> {
    if (this.confirmationForm.invalid || this.isSubmitting()) {
      this.confirmationForm.markAllAsTouched();
      return;
    }

    this.isSubmitting.set(true);
    this.errorMessage.set('');

    try {
      await this.auth.confirmSignUp(
        this.confirmationEmail(),
        this.confirmationForm.controls.code.value.trim(),
      );
      await this.router.navigate(['/login'], { queryParams: { signup: 'confirmed' } });
    } catch (error) {
      this.errorMessage.set(
        error instanceof Error ? error.message : 'Cognito could not confirm your account.',
      );
    } finally {
      this.isSubmitting.set(false);
    }
  }

  protected async resendCode(): Promise<void> {
    if (this.isResending() || this.isSubmitting()) return;

    this.isResending.set(true);
    this.errorMessage.set('');
    this.resendMessage.set('');

    try {
      const destination = await this.auth.resendConfirmationCode(this.confirmationEmail());
      if (destination) this.deliveryDestination.set(destination);
      this.resendMessage.set('A new confirmation code has been sent.');
    } catch (error) {
      this.errorMessage.set(
        error instanceof Error ? error.message : 'Cognito could not resend the code.',
      );
    } finally {
      this.isResending.set(false);
    }
  }

  protected useExistingCode(): void {
    const emailControl = this.detailsForm.controls.email;
    if (emailControl.invalid) {
      emailControl.markAsTouched();
      return;
    }

    const email = emailControl.value.trim();
    this.confirmationEmail.set(email);
    this.deliveryDestination.set(email);
    this.errorMessage.set('');
  }

  protected changeEmail(): void {
    if (this.isSubmitting() || this.isResending()) return;
    this.confirmationEmail.set('');
    this.deliveryDestination.set('');
    this.confirmationForm.reset();
    this.errorMessage.set('');
    this.resendMessage.set('');
  }
}
