#include <vector>

class PinesAlgorithm
{
private:
    /* data */
public:
    std::vector<std::vector<double> > cBar, sBar;

    double s, t, u;
    double a, mu;
    int N; // Degree of coefficient model
    int P;  // Number of measurements

    std::vector<double> rhol;
    std::vector<double> rE, iM;
    std::vector<std::vector<double> > aBar;
    std::vector<std::vector<double> > n1, n2, n1q, n2q;

    std::vector<double> compute_acc(std::vector<double>, std::vector<std::vector<double> >, std::vector<std::vector<double> >);
    int print_percentage(int n, int size, int progress);
    PinesAlgorithm(double r0, double muBdy, int degree);
    ~PinesAlgorithm();

    double getK(const unsigned int degree) {return ((degree == 0) ? 1.0 : 2.0); }
};
