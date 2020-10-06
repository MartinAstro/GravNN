#include <vector>
#include <Eigen/Dense>
#include "Eigen/Sparse"
#include "Eigen/SparseCore"
#include "Eigen/SparseQR"
#include "Eigen/OrderingMethods"
struct Coefficients
{
    std::vector<std::vector<double> > C_lm;
    std::vector<std::vector<double> > S_lm;
};

class Regression
{
public:
    Regression(std::vector<double>, std::vector<double>, int, double, double);
    ~Regression();
    Coefficients coef_regress;
    Coefficients perform_regression(int, double);
    // Result variables
    bool solution_exists; // boolean determines if solution was regressed
    bool solution_unique; // boolean determining if sufficient measurements to guarenttee uniqueness
    

private:

    Eigen::VectorXd pos_meas_eigen; // 1D position Eigen::VectorXd
    Eigen::VectorXd acc_meas_eigen; // 1D acceleration Eigen::VectorXd
    std::vector<double> pos_meas; //1D position std::vector<double>
    std::vector<double> acc_meas; //1D acceleration std::vector<double>


    double s, t, u;
    double a; // reference radius of body
    double mu; // mu of body
    int N; // Max degree (l) to regress
    int P;  // Number of measurements
    int l_i; // First degree to be regressed 
    int M_total;  // Number of coefficients to skip
    int M_skip; // Total number of coefficients up to degree l_{f+1} -- e.g if l_max = 2 then M_total = 12
    
    // Pines variables
    std::vector<double> r, i;
    std::vector<std::vector<double> > A;
    std::vector<std::vector<double> > n1, n2;
    std::vector<double> rho;

    // Regression variables
    double getK(int l) 
    {
        return (l==0) ? 1.0 : 2.0;
    };

    void set_regression_variables(double, double, double);
    void find_solution(double, Eigen::MatrixXd &, Eigen::SparseMatrix<double> &);
    void format_coefficients(Eigen::VectorXd coef_list);


};
