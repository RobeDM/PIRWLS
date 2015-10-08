
double kernel(svm_dataset dataset, int index1, int index2, properties props){

    double sum = 0.0;   
    
    svm_sample *x=dataset.x[index1];
    svm_sample *y=dataset.x[index2];

    if (dataset.sparse==0){
        int maxdim = dataset.maxdim;
        int i;
        for(i=0;i<maxdim;i++){
	    sum+=pow((x->value) - (y->value),2);
	    ++x;
	    ++y;
	}

    }else{

        sum += (dataset.quadratic_value[index1])+(dataset.quadratic_value[index2]);
        while(x->index !=-1 && y->index !=-1) {
            if(x->index == y->index){
                sum += -2.0 * (x->value) * (y->value);
                ++x;
                ++y;
            }else{
                if((x->index) < (y->index)){
                    ++x;
                }else{
                    ++y;
                }
            }
        }
    }

    return exp(-(props.Kgamma)*sum);
}



double kernelTest(svm_dataset dataset, int index1, model mymodel, int index2){

    double sum = 0.0;

    svm_sample *x=dataset.x[index1];
    svm_sample *y=mymodel.x[index2];

    if (mymodel.sparse==0){
        int maxdim = mymodel.maxdim;
        int i;
        for(i=0;i<maxdim;i++){
            sum+=pow((x->value)-(y->value) ,2);
            ++x;
            ++y;
        }
    }else{

        sum += (dataset.quadratic_value[index1])+(mymodel.quadratic_value[index2]);
        while(x->index !=-1 && y->index !=-1) {
            if(x->index == y->index){
                sum += -2.0 * (x->value) * (y->value);
                ++x;
                ++y;
            }else{
                if((x->index) < (y->index)){
                    ++x;
                }else{
                    ++y;
                }
            }
        }
    }

    return exp(-(mymodel.Kgamma)*sum);
}

