import React from 'react';
import { cx } from './classNames';

const elevationShadow = {
  0: 'shadow-none',
  1: 'shadow-sm',
  2: 'shadow',
  3: 'shadow-lg',
};

export function Card({ elevation = 1, className, children, hoverable, ...props }){
  return (
    <div className={cx('bg-surface text-text border border-border rounded-md', elevationShadow[elevation] || elevationShadow[1], hoverable && 'transition hover:shadow-lg hover:-translate-y-[1px]', className)} {...props}>
      {children}
    </div>
  );
}

export function CardHeader({ className, children }){
  return <div className={cx('px-4 py-3 border-b border-border font-semibold', className)}>{children}</div>;
}

export function CardBody({ className, children }){
  return <div className={cx('px-4 py-3', className)}>{children}</div>;
}

export function CardFooter({ className, children }){
  return <div className={cx('px-4 py-3 border-t border-border', className)}>{children}</div>;
}

