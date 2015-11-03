void initMemory(int Threads, int size);
void updateMemory(int Threads, int size);
void ParallelChol(double *matrix,int r,int c, int ro, int co, int n,int nCores, int deep);
void Chol(double *matrix,int r,int c, int ro, int co, int n,int nCores,int numTh, int deep,int posIni,double *memaux, int blockSize);
void ParallelLinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, int nCores);
void LinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, int nCores, int numTh,int posIni,double *memaux, int blockSize);
void DiagInversion(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int numTh);
void InversionNLProducts(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int deep,int numTh, double *memaux, int blockSize);
void InversionLNProducts(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int deep,int numTh, double *memaux, int blockSize);
void TriangleInversion(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni, int numTh,double *memaux,int blockSize);
void NNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh,int orientation);
void MoveMatrix(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2, int n1, int n2, int nCores,int posIni,int numTh);
void TNNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);
void NNTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);
void AATProduct(double *m1,int r1,int ro1,int c1, int co1,int n1,int n2,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);
void LNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);
void LTNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);
void NLProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);
void NLTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);
void getSubMatrix(double *matrix,int size1,int size2,int O1,int O2,double *A, int size3,int size4,int nCores);
void putSubMatrix(double *matrix,int size1,int size2,int O1,int O2,double *A, int size3,int size4,int nCores);




