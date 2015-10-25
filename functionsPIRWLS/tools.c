
typedef struct properties{
    double Kgamma;
    double C;
    int Threads;
    int MaxSize;
}properties;

typedef struct predictProperties{
    int Labels;
    int Threads;
}predictProperties;

typedef struct model{
    double Kgamma;
    int sparse;
    int nSVs;
    int nElem;
    double *weights;    
    struct svm_sample **x;
    double *quadratic_value;    
    int maxdim;
    double bias;
}model;

typedef struct svm_sample{
    int index;
    double value;
}svm_sample;

typedef struct svm_dataset{
    int l;
    int sparse;
    int maxdim;
    double *y;
    struct svm_sample **x;
    double *quadratic_value;
}svm_dataset;

static int compare (const void * a, const void * b){
  if (*(double*)a < *(double*)b) return -1;
  else if (*(double*)a > *(double*)b) return +1;
  else return 0;  
}

void printInstructions() {
    fprintf(stderr, "PIRWLS-train: This software train the SVM on the given training set and ");
    fprintf(stderr, "generages a model for futures prediction use.\n\n");
    fprintf(stderr, "Usage: PIRWLS-train [options] training_set_file model_file\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -g gamma: set gamma in radial basis kernel function (default 1)\n");
    fprintf(stderr, "       radial basis K(u,v)= exp(-gamma*|u-v|^2)\n");
    fprintf(stderr, "  -c Cost: set SVM Cost (default 1)\n");
    fprintf(stderr, "  -t Threads: Number of threads (default 1)\n");
    fprintf(stderr, "  -w Working set size: Size of the Least Squares problem in every iteration (default 500)\n");
}


void printInstructionsPredict() {
    fprintf(stderr, "PIRWLS-predict: This software predicts the label of a SVM given a data set of samples and a model obtained with PIRWLS-train");
    fprintf(stderr, "and store the results in an output file.\n\n");
    fprintf(stderr, "Usage: PIRWLS-predict [options] data_set_file model_file output_file\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t Number of Threads: (default 1)\n");
    fprintf(stderr, "  -l type of data set: (default 0)\n");
    fprintf(stderr, "       0 -- Data set with no target as first dimension.\n");
    fprintf(stderr, "       1 -- Data set with label as first dimension (obtains accuracy too)\n");
    fprintf(stderr, "Note:\n");
    fprintf(stderr, "       The data set file must have the same format as the data set\n");
    fprintf(stderr, "       given to PIRWLS-train.\n");
}

properties TrainParameters(int* argc, char*** argv) {

    properties props;
    props.Kgamma = 1.0;
    props.C = 1.0;
    props.Threads=1;
    props.MaxSize=500;

    int i,j;
    for (i = 1; i < *argc; ++i) {
        if ((*argv)[i][0] != '-') break;
        if (++i >= *argc) {
            printInstructions();
            exit(1);
        }

        char* param_name = &(*argv)[i-1][1];
        char* param_value = (*argv)[i];
        if (strcmp(param_name, "g") == 0) {    	
            props.Kgamma = atof(param_value);
        } else if (strcmp(param_name, "c") == 0) {
            props.C = atof(param_value);
        } else if (strcmp(param_name, "t") == 0) {
            props.Threads = atoi(param_value);
        } else if (strcmp(param_name, "w") == 0) {
            props.MaxSize = atoi(param_value);
        } else {
            fprintf(stderr, "Unknown parameter %s\n",param_name);
            printInstructions();
            exit(2);
        }
    }
  
    for (j = 1; i + j - 1 < *argc; ++j) {
        (*argv)[j] = (*argv)[i + j - 1];
    }
    *argc -= i - 1;

    return props;

}

predictProperties PredictParameters(int* argc, char*** argv) {

    predictProperties props;
    props.Labels=0;
    props.Threads=1;
	
	
    int i;
    for (i = 1; i < *argc; ++i) {
        if ((*argv)[i][0] != '-') break;
        if (++i >= *argc) {
            printInstructionsPredict();
            exit(1);
        }

        char* param_name = &(*argv)[i-1][1];
        char* param_value = (*argv)[i];
        
        if (strcmp(param_name, "t") == 0) {    	
            props.Threads = atof(param_value);
        } else if (strcmp(param_name, "l") == 0) {
            props.Labels = atoi(param_value);
            if(props.Labels !=0 && props.Labels !=1){
      	        printf("\nInvalid type of test data set:%d\n",props.Labels);
                exit(2);
            }
        } else {
            fprintf(stderr, "Unknown parameter %s\n",param_name);
            printInstructionsPredict();
            exit(2);
        }
    }
	  int j;
    for (j = 1; i + j - 1 < *argc; ++j) {
        (*argv)[j] = (*argv)[i + j - 1];
    }
    *argc -= i - 1;
    
    return props;
}


svm_dataset readTrainFile(char filename[]){

    svm_dataset dataset;
	
    int arraysize=256;
  	
    if (filename == NULL){
        fprintf(stderr, "File not specified");
        exit(2);
    }
		
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "File not found: %s\n",filename);
        exit(2);
    }	

    char fileline[100000];

    dataset.l = 0;
    int elements = 0;
    dataset.sparse = 0;

    while (fgets(fileline, 100000, file) != NULL){
        char *p = strtok(fileline," \t");


        while(1){
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') break;
            ++elements;
        }
        ++elements;
        ++dataset.l;
    }

    rewind(file);
    
    dataset.y = (double *) calloc(dataset.l,sizeof(double));
    dataset.quadratic_value = (double *) calloc(dataset.l,sizeof(double));
    dataset.x = (svm_sample **) calloc(dataset.l,sizeof(svm_sample *));
    svm_sample* features = (svm_sample *) calloc(elements,sizeof(svm_sample));
    dataset.maxdim=0;

    int max_index = 0;
    int i,j,dm=0;
    char *endptr;
    char *idx, *val, *label;
    int inst_max_index;
    int errno;


    for(i=0;i<dataset.l;i++){

        inst_max_index = -1;
        if (fgets(fileline, 100000, file)== NULL){
            fprintf(stderr, "Error reading data file\n");
            exit(2);
        }

        dataset.x[i] = &features[j];
	      label = strtok(fileline," \t\n");

        if(label == NULL){
            fprintf(stderr, "Wrong file format\n");
            exit(2);
        }

        dataset.y[i] = strtod(label,&endptr);

        if(endptr == label || *endptr != '\0'){
            fprintf(stderr, "Wrong file format\n");
            exit(2);
        }
        dm = 0;
        while(1){
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL) break;

            errno = 0;
            features[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || features[j].index <= inst_max_index){
                fprintf(stderr, "Wrong file format\n");
                exit(2);
            }else{
                inst_max_index = features[j].index;
            }
            if(features[dm].index != features[j].index){
                dataset.sparse=1;
            }
            errno = 0;
            features[j].value = strtod(val,&endptr);
            dataset.quadratic_value[i] += pow(strtod(val,&endptr),2);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))){
                fprintf(stderr, "Wrong file format\n");
                exit(2);
            }
            ++dm;
            ++j;
        }

        if(inst_max_index > max_index){
            max_index = inst_max_index;
        }
        features[j++].index = -1;

    }

    dataset.maxdim=max_index;
    fclose(file);
    return dataset;

}

svm_dataset readUnlabeledFile(char filename[]){

    svm_dataset dataset;
	
    int arraysize=256;
  	
    if (filename == NULL){
        fprintf(stderr, "File not specified");
        exit(2);
    }
		
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "File not found: %s\n",filename);
        exit(2);
    }	

    char fileline[100000];

    dataset.l = 0;
    int elements = 0;
    dataset.sparse = 0;

    while (fgets(fileline, 100000, file) != NULL){
        char *p;

        while(1){
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') break;
            ++elements;
        }
        ++elements;
        ++dataset.l;
    }

    rewind(file);
    
    dataset.y = (double *) calloc(dataset.l,sizeof(double));
    dataset.quadratic_value = (double *) calloc(dataset.l,sizeof(double));
    dataset.x = (svm_sample **) calloc(dataset.l,sizeof(svm_sample *));
    svm_sample* features = (svm_sample *) calloc(elements,sizeof(svm_sample));
    dataset.maxdim=0;

    int max_index = 0;
    int i,j,dm=0;
    char *endptr;
    char *idx, *val, *label;
    int inst_max_index;
    int errno;


    for(i=0;i<dataset.l;i++){

        inst_max_index = -1;
        if (fgets(fileline, 100000, file)== NULL){
            fprintf(stderr, "Error reading data file\n");
            exit(2);
        }

        dataset.x[i] = &features[j];
        dm = 0;
        while(1){
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL) break;

            errno = 0;
            features[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || features[j].index <= inst_max_index){
                fprintf(stderr, "Wrong file format\n");
                exit(2);
            }else{
                inst_max_index = features[j].index;
            }
            if(features[dm].index != features[j].index){
                dataset.sparse=1;
            }
            errno = 0;
            features[j].value = strtod(val,&endptr);
            dataset.quadratic_value[i] += pow(strtod(val,&endptr),2);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))){
                fprintf(stderr, "Wrong file format\n");
                exit(2);
            }
            ++dm;
            ++j;
        }

        if(inst_max_index > max_index){
            max_index = inst_max_index;
        }
        features[j++].index = -1;

    }

    dataset.maxdim=max_index;
    fclose(file);
    return dataset;

}

void storeModel(model * mod, FILE *Output){

    int aux;
    aux=fwrite(&mod->Kgamma, sizeof(double), 1, Output);    
    aux=fwrite(&mod->bias, sizeof(double), 1, Output);
    aux=fwrite(&mod->maxdim, sizeof(int), 1, Output);
    aux=fwrite(&mod->sparse, sizeof(int), 1, Output);
    aux=fwrite(&mod->nSVs, sizeof(int), 1, Output);
    aux=fwrite(&mod->nElem, sizeof(int), 1, Output);
    aux=fwrite(mod->weights, (mod->nSVs)*sizeof(double), 1, Output);
    aux=fwrite(mod->quadratic_value, (mod->nSVs)*sizeof(double), 1, Output);
    aux=fwrite(mod->x[0], (mod->nElem)*sizeof(svm_sample), 1, Output);

}

void readModel(model * mod, FILE *Input){
	
    int aux;
    aux=fread(&mod->Kgamma, sizeof(double), 1, Input);
    aux=fread(&mod->bias, sizeof(double), 1, Input);
    aux=fread(&mod->maxdim, sizeof(int), 1, Input);
    aux=fread(&mod->sparse, sizeof(int), 1, Input);
    aux=fread(&mod->nSVs, sizeof(int), 1, Input);    
    aux=fread(&mod->nElem, sizeof(int), 1, Input);    
    mod->weights = (double *)malloc((mod->nSVs)*sizeof(double));
    mod->quadratic_value = (double *)malloc((mod->nSVs)*sizeof(double));    
    aux=fread(mod->weights, (mod->nSVs)*sizeof(double), 1, Input);
    aux=fread(mod->quadratic_value, (mod->nSVs)*sizeof(double), 1, Input);
    mod->x = (svm_sample **)malloc((mod->nSVs)*sizeof(svm_sample *));    
    svm_sample* features = (svm_sample *) calloc((mod->nElem),sizeof(svm_sample));    
    aux=fread(features, (mod->nElem)*sizeof(svm_sample), 1, Input);

    mod->x[0]=&features[0];    
    int iterSV=1;
    for(aux=0;aux<(mod->nElem);aux++){
        if (features[aux].index == -1){
            if(iterSV<mod->nSVs) mod->x[iterSV]=&features[aux+1];
            ++iterSV;
        }
    }
}

model calculateModel(properties props, svm_dataset dataset, double * beta ){
    model classifier;
    classifier.Kgamma = props.Kgamma;
    classifier.bias = beta[dataset.l];
    classifier.sparse = dataset.sparse;
    classifier.maxdim = dataset.maxdim;
    
    int nElem=0;
    int nSVs=0;
    svm_sample *iteratorSample;
    svm_sample *classifierSample;
    int i;
    for (i =0;i<dataset.l;i++){
        if(beta[i] != 0.0){
            ++nSVs;
            iteratorSample = dataset.x[i];
            while (iteratorSample->index != -1){
            	  ++iteratorSample;
                ++nElem;
            }
            ++nElem;
        }
    }    

    classifier.nSVs = nSVs;
    classifier.nElem = nElem;
    classifier.weights = (double *) calloc(nSVs,sizeof(double));
    classifier.quadratic_value = (double *) calloc(nSVs,sizeof(double));

    classifier.x = (svm_sample **) calloc(nSVs,sizeof(svm_sample *));
    svm_sample* features = (svm_sample *) calloc(nElem,sizeof(svm_sample));

    
    int indexIt=0;
    int featureIt=0;
    for (i =0;i<dataset.l;i++){
        if(beta[i] != 0.0){
            classifier.quadratic_value[indexIt]=dataset.quadratic_value[i];
            classifier.weights[indexIt]=beta[i];            
            classifier.x[indexIt] = &features[featureIt];
            
            iteratorSample = dataset.x[i];
            classifierSample = classifier.x[indexIt];

            while (iteratorSample->index != -1){
                classifierSample->index = iteratorSample->index;
                classifierSample->value = iteratorSample->value;
                ++classifierSample;
                ++iteratorSample;
                ++featureIt;
            }

            classifierSample->index = iteratorSample->index;
            
            indexIt++;
            ++featureIt;
        }
    }        
    return classifier;
}
