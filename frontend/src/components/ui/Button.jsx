import React from 'react';
import { cx } from './classNames';

const base = 'inline-flex items-center justify-center font-medium rounded-md transition-transform duration-fast focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-60 disabled:cursor-not-allowed';

const variants = {
  primary: 'bg-primary text-on-primary hover:brightness-110 active:scale-[0.98] focus:ring-primary',
  secondary: 'bg-secondary text-on-primary hover:brightness-110 active:scale-[0.98] focus:ring-secondary',
  ghost: 'bg-transparent text-text hover:bg-muted active:scale-[0.98] focus:ring-border',
  danger: 'bg-danger text-on-primary hover:brightness-110 active:scale-[0.98] focus:ring-danger',
};

const sizes = {
  sm: 'px-2 py-1 text-sm',
  md: 'px-4 py-2 text-sm',
  lg: 'px-5 py-3 text-base',
};

export function Button({
  as: Comp = 'button',
  variant = 'primary',
  size = 'md',
  className,
  children,
  loading,
  leftIcon,
  rightIcon,
  ...props
}){
  return (
    <Comp className={cx(base, variants[variant], sizes[size], className)} {...props}>
      {leftIcon && <span className="mr-2 inline-flex">{leftIcon}</span>}
      {loading && <span className="mr-2 inline-block w-4 h-4 border-2 border-on-primary/70 border-t-transparent rounded-full animate-spin"></span>}
      {children}
      {rightIcon && <span className="ml-2 inline-flex">{rightIcon}</span>}
    </Comp>
  );
}

