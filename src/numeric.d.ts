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
