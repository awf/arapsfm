#ifndef __OPTIMISER_OPTIONS_H__
#define __OPTIMISER_OPTIONS_H__

struct OptimiserOptions
{
    int maxIterations;
    int minIterations;
    double tau;
    double lambda;
    double gradientThreshold;
    double updateThreshold;
    double improvementThreshold;
    bool useAsymmetricLambda;
    int verbosenessLevel;
};

#endif
