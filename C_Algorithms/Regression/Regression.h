#include <vector>
#include <Eigen/Dense>

class Regression
{
private:
    /* data */
public:
    Eigen::VectorXd coeff;
    Eigen::VectorXd positions;
    Eigen::VectorXd accelerations;





    double s, t, u;
    double a, mu;
    int N; // Degree of coefficient model
    int P;  // Number of measurements

    std::vector<double> r, i;
    std::vector<std::vector<double> > A;
    std::vector<std::vector<double> > n1, n2;

    std::vector<double> rho;

    double getK(int l) 
    {
        return (l==0) ? 1.0 : 2.0;
    };

    void perform_regression();
    void populate_variables(double, double, double);


    Regression(std::vector<double>, std::vector<double>, int, double, double);
    //Regression(std::vector<std::vector<double>>, std::vector<std::vector<double>>, int);
    ~Regression();
};
