void LinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, MKL_INT nCores, MKL_INT numTh,int posIni,double *memaux, int blockSize);
void TriangleInversion(double *matrix,int r, int c, int ro, int co, MKL_INT n,MKL_INT nCores,int posIni, int numTh,double *memaux,int blockSize);
void ParallelLinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, MKL_INT nCores);
void Chol(double *matrix,int r,int c, int ro, int co, MKL_INT n,MKL_INT nCores,MKL_INT numTh, MKL_INT deep,int posIni,double *memaux, int blockSize);
void InversionNLProducts(double *matrix,int r, int c, int ro, int co, MKL_INT n,MKL_INT nCores,MKL_INT posIni,MKL_INT deep,MKL_INT numTh, double *memaux, int blockSize);
void InversionLNProducts(double *matrix,int r, int c, int ro, int co, MKL_INT n,MKL_INT nCores,MKL_INT posIni,MKL_INT deep,MKL_INT numTh, double *memaux, int blockSize);
void Inversion(double *matrix, MKL_INT r,MKL_INT c,MKL_INT ro,MKL_INT co, MKL_INT n, double *result, MKL_INT rr,MKL_INT cr,MKL_INT ror,MKL_INT cor, MKL_INT nCores,MKL_INT posIni,MKL_INT numTh, int deep,double *memaux, int blockSize,double *meminv);


void ParallelInversion(double *matrix, MKL_INT r,MKL_INT c,MKL_INT ro,MKL_INT co, MKL_INT n, double *result, MKL_INT rr,MKL_INT cr,MKL_INT ror,MKL_INT cor, MKL_INT nCores,MKL_INT posIni);
void MoveMatrix(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2, MKL_INT n1, MKL_INT n2, MKL_INT nCores,MKL_INT posIni,MKL_INT numTh);
void printMatCol(double *m,int n1,int n2);
void ParallelNNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,MKL_INT n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni,MKL_INT orientation);
void ParallelTNNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,MKL_INT n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni);
void ParallelNNTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,MKL_INT n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni);
void ParallelAATProduct(double *m1,int r1,int ro1,int c1, int co1,MKL_INT n1,MKL_INT n2,double K1,double K2,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni);
void ParallelLNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,double K1,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni);
void ParallelLTNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,double K1,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni);
void ParallelNLProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,double K1,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni);
void ParallelNLTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,double K1,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni);
void NNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,MKL_INT n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni,MKL_INT numTh,MKL_INT orientation);
void TNNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,MKL_INT n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni,MKL_INT numTh);
void NNTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,MKL_INT n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni,MKL_INT numTh);
void AATProduct(double *m1,int r1,int ro1,int c1, int co1,MKL_INT n1,MKL_INT n2,double K1,double K2,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni,MKL_INT numTh);
void LNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,double K1,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni,MKL_INT numTh);
void LTNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,double K1,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni,MKL_INT numTh);
void NLProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,double K1,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni,MKL_INT numTh);
void NLTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,MKL_INT n1,MKL_INT n2,double K1,double *result,int rr,int ror,int cr, int cor, MKL_INT nCores,MKL_INT posIni,MKL_INT numTh);


void putSubMatrix(double *matrix,MKL_INT size1,MKL_INT size2,MKL_INT O1,MKL_INT O2,double *A, MKL_INT size3,MKL_INT size4,int nCores);
void getSubMatrix(double *matrix,MKL_INT size1,MKL_INT size2,MKL_INT O1,MKL_INT O2,double *A, MKL_INT size3,MKL_INT size4,int nCores);
void ParallelCholesky(double *matrix,int r,int c, int ro, int co, MKL_INT n,MKL_INT nCores, MKL_INT deep);
void ParallelTriangleInversion(double *matrix,int r, int c, int ro, int co, MKL_INT n,MKL_INT nCores);
void DecColOrder(double *matrix,MKL_INT size1,MKL_INT size2,MKL_INT size3,MKL_INT size4,double *A, double *B, double *C, double *D,int nCores);
void CompColOrder(double *matrix,MKL_INT size1,MKL_INT size2,MKL_INT size3,MKL_INT size4,double *A, double *B, double *C, double *D,int nCores);
