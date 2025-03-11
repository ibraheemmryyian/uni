export const FAQ_CATEGORIES = {
  GENERAL_INQUIRY: 'General inquiry',
  TIME_EXTENSION: 'Time extension',
  DISPUTE: 'Dispute',
  PAYMENT_METHOD: 'Payment method',
  PAYMENT_PLAN: 'Payment plan',
  PAYMENT_ISSUE: 'Payment method issue - lender',
  PAID: 'Paid'
} as const;

export type FAQCategory = typeof FAQ_CATEGORIES[keyof typeof FAQ_CATEGORIES];