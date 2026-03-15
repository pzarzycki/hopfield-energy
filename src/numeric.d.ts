declare module "numeric" {
  interface NumericSvdResult {
    U: number[][];
    S: number[];
    V: number[][];
  }

  interface NumericModule {
    inv(matrix: number[][]): number[][];
    svd(matrix: number[][]): NumericSvdResult;
  }

  const numeric: NumericModule;
  export default numeric;
}

declare module "react-katex" {
  import type { ReactNode } from "react";

  export function BlockMath(props: { math?: string; children?: ReactNode }): ReactNode;
  export function InlineMath(props: { math?: string; children?: ReactNode }): ReactNode;
}
