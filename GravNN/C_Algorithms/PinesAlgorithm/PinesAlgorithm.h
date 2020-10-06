#include <vector>

class PinesAlgorithm
{
private:
    /* data */
public:
   
    double s, t, u;
    double a, mu;
    int N; // Degree of coefficient model
    int P;  // Number of measurements

    std::vector<std::vector<double> > n1, n2, n1q, n2q;
    std::vector<std::vector<double> > cBar, sBar;

    std::vector<double> compute_acc(std::vector<double>);
    std::vector<double> compute_acc_components(std::vector<double>);
    std::vector<double> compute_acc_thread(double, double, double);

    int print_percentage(int n, int size, int progress);
    PinesAlgorithm(double r0, double muBdy, int degree, std::vector<std::vector<double> >, std::vector<std::vector<double> >);
    ~PinesAlgorithm();

    double getK(const unsigned int degree) {return ((degree == 0) ? 1.0 : 2.0); }
};
