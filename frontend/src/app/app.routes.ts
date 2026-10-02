import { Routes } from '@angular/router';

import { signedOutGuard } from './auth/auth.guard';
import { Chat } from './chat/chat';
import { Login } from './login/login';
import { Signup } from './signup/signup';

export const routes: Routes = [
  {
    path: 'login',
    component: Login,
    canActivate: [signedOutGuard],
    title: 'Sign in — Membership Atlas',
  },
  {
    path: 'signup',
    component: Signup,
    canActivate: [signedOutGuard],
    title: 'Create account — Membership Atlas',
  },
  {
    path: '',
    component: Chat,
    title: 'Oriana — Membership Atlas assistant',
  },
  { path: '**', redirectTo: '' },
];
