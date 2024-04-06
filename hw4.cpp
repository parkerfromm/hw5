#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <armadillo>
#include <cmath>
#include <string>
#include <utility>
#include <map>
using namespace std;
// using namespace arma;
enum electrons
{
    alpha_,
    beta_
};

struct Function
{
    int atomic_num;
    vector<double> R;
    vector<double> norms;
    vector<double> exponets;
    vector<double> coefs;
    vector<int> orbital;
    double B;
    double I_A;
    int val_elec;
    int l;
    int m;
    int n;
    double int_pot; // iniozation potential ?? soemthing like that

    /// aaaa need to update based on other map
    // Function()
    //     : atomic_num(0), // Initialize to default value (0 for an integer)
    //       R({}),         // Empty vector
    //       norms({}),
    //       exponets({}),
    //       coefs({}),
    //       orbital({}),
    //       B(0.0),      // Initialize to default value (0.0 for a double)
    //       I_A(0.0),    // Initialize to default value (0.0 for a double)
    //       val_elec(0), // Initialize to default value (0 for an integer)
    //       l(0),        // Initialize to default value (0 for an integer)
    //       m(0),        // Initialize to default value (0 for an integer)
    //       n(0),        // Initialize to default value (0 for an integer)
    //       int_pot(0.0) // Initialize to default value (0.0 for a double)
    // {
    // }

    Function(int atomic_num, vector<double> R, vector<double> norms, vector<double> exponets, vector<double> coefs, vector<int> orbital,
             double B, double I_A, int val_elec, int l, int m, int n, double int_pot)
        : atomic_num(atomic_num), R(R), norms(norms), exponets(exponets), coefs(coefs), orbital(orbital),
          B(B), I_A(I_A), val_elec(val_elec), l(l), m(m), n(n), int_pot(int_pot)
    {
    }
};

/// int is atomic number
// vector<vector<double>> is a vector that first componet will be the coefficants, second will be s, orbitial
// third will be p orbitial if applicable and so on.

// map < int, vector<vector<double, 3>> map;

// vector < pair<int, vector<double>> atoms;
// map<int, Function> atoms;

// basis fucntion centered on U and V

class CNDO2
{
public:
    CNDO2(string file_name);
    // double p_uv(vector<double> C_U, vector<double> C_V, electrons spin);
    // double p_uv_tot(vector<double> C_U, vector<double> C_V);

    double p_uv(arma::Col<double> C_U, arma::Col<double> C_V, electrons spin);
    double p_uv_tot(int U, int V);

    double P_tot(Function &fun);
    // double calc_gamma_AB();
    double gamma_AB(Function &funA, Function &funB);
    double gamma_AA(Function &funA, Function &funB);
    void make_gamma();
    void Suv_RA();
    void gamma_RA();
    void update_G();
    void make_Xuv();
    void make_yAB();

    // need to determine valance electrons from the charge ( p and q maybe they are VE)

    void calc_P();
    void calc_f();
    // void calc_C(arma::Mat<double> &C, arma::Mat<double> &f);
    void calc_C_and_e();
    void calc_S();
    double inf_norm(arma::Mat<double> &P_new, arma::Mat<double> &P_old);
    void huckle(arma::Mat<double> &H);
    // do not change or code will break
    double total_energy();
    int N;
    pair<double, double> one_step();
    void run();
    void info();
    double yBA();
    double yAB();

private:
    void read_basis(string filepath, int atomic_num);
    void make_map();
    void add_norms();
    void calc_basis();
    void get_s_basis();
    void read_xyz(string filepath);

    double overlap_integral_1d_analytical(double X_A, double X_B, int l_A, int l_B, double aalpha, double bbeta);
    int binomial_coefficient(int m, int n);
    double X_p(double X_A, double X_B, double aalpha, double bbeta);
    int double_factorial(int num);
    int factorial(int num);

    int alpha;
    int beta;
    vector<pair<int, vector<double>>> atoms;
    vector<Function> basis;
    vector<Function> s_basis;
    // represent the AOs as the rows
    // columns will represent a MO then

    int p;
    int q;
    // double gamma_AB;
    arma::Mat<double> core_H;
    arma::Mat<double> S;
    arma::Mat<double> G;
    arma::Mat<double> Ga;
    arma::Mat<double> Gb;
    arma::Mat<double> X;
    arma::Mat<double> F_alpha;
    arma::Mat<double> F_beta;
    arma::Mat<double> P_alpha;
    arma::Mat<double> P_beta;
    arma::Mat<double> C_alpha;
    arma::Mat<double> C_beta;
    arma::Col<double> e_alpha;
    arma::Col<double> e_beta;
    arma::Col<double> f_alpha;
    arma::Col<double> f_beta;
    arma::Mat<double> S_ra;
    arma::Mat<double> S_rx;
    arma::Mat<double> S_ry;
    arma::Mat<double> S_rz;
    arma::Mat<double> gradient;
    vector<vector<Function>> A_basis;

    double alpha_change;

    double beta_change;

    int atom_A;
    int atom_B;
    int num_atoms;

    map<int, vector<vector<double>>> basis_map;
    map<int, vector<double>> params_map;
    map<int, int> electron_map;

    // init both still
    // gamma
    double P_AA;
    double P_BB;
};

CNDO2::CNDO2(string file_name)
{

    read_xyz(file_name);
    int a = 0;
    int b = 0;
    int total_val = 0;
    atom_A = 0;
    atom_B = 0;

    for (auto &atom : atoms)
    {
        if (atom.first == 1)
        {
            b++;
            total_val += 1;
        }
        else
        {
            a++;

            // lazy in theroy will only work for elements
            // atomic number 3-10 otherwise need to change code (fine for assignment)
            total_val += (atom.first - 2);
        }
    }

    N = 4 * a + b;

    alpha = total_val / 2;
    beta = alpha;
    if (total_val % 2 != 0)
    {
        alpha++;
    }

    // in order -B, 0.5(I_s + A_s), 0.5(I_p + A_p)
    params_map[1] = {9, 7.176, 0};
    params_map[6] = {21, 14.051, 5.572};
    params_map[7] = {25, 19.316, 7.275};
    params_map[8] = {31, 25.390, 9.111};
    params_map[9] = {39, 32.272, 11.080};

    electron_map[1] = 1;
    electron_map[6] = 4;
    electron_map[7] = 5;
    electron_map[8] = 6;
    electron_map[9] = 7;

    core_H.set_size(N, N);
    S.set_size(N, N);
    F_alpha.set_size(N, N);
    P_alpha.set_size(N, N);
    C_alpha.set_size(N, N);
    F_beta.set_size(N, N);
    Ga.set_size(N, N);
    Gb.set_size(N, N);
    P_beta.set_size(N, N);
    C_beta.set_size(N, N);
    S_ra.set_size(3, pow(N, 2));
    S_rx.set_size(N, N);
    S_ry.set_size(N, N);
    S_rz.set_size(N, N);
    gradient.set_size(3, num_atoms);

    X.set_size(N, N);

    // initiate C = zero matrix
    make_map();
    calc_basis();
    add_norms();
    // get_s_basis();

    // G.set_size(s_basis.size(), s_basis.size());
    G.set_size(2, 2);

    calc_S();
    make_gamma();
    huckle(core_H);

    P_alpha.zeros();
    P_beta.zeros();
    F_alpha = core_H;
    F_beta = core_H;
    alpha_change = 1;
    beta_change = 1;
    // P_AA = P_tot(atom_A);
    // P_BB = P_tot(atom_B);
}

void CNDO2::read_xyz(string filepath)
{

    ifstream myfile(filepath);
    int charge;

    if (myfile.is_open())
    {
        myfile >> num_atoms;
        myfile >> charge;

        int atomic_num;
        vector<double> coords(3);

        for (int i = 0; i < num_atoms; i++)
        {

            myfile >> atomic_num;
            myfile >> coords[0];
            myfile >> coords[1];
            myfile >> coords[2];

            // pair<int, vector<double,3>> atom = make_pair(atomic_num, )

            atoms.emplace_back(atomic_num, coords);

            if (atom_A == 0 && atom_B == 0)
            {
                atom_A = atomic_num;
            }
            else if (atom_B == 0)
            {
                atom_B = atomic_num;
            }
        }
    }

    myfile.close();
}

void CNDO2::read_basis(string filepath, int atomic_num)
{
    vector<vector<double>> return_vec;
    vector<double> exponets;

    fstream myfile(filepath);

    if (myfile.is_open())
    {
        double exp;
        double coef;

        if (atomic_num == 1)
        {
            vector<double> coefs;

            for (int i = 0; i < 3; i++)
            {
                myfile >> exp;
                myfile >> coef;
                exponets.push_back(exp);
                coefs.push_back(coef);
            }
            return_vec.push_back(exponets);
            return_vec.push_back(coefs);
        }
        else
        {
            vector<double> coefs_s;
            vector<double> coefs_p;

            for (int i = 0; i < 3; i++)
            {

                myfile >> exp;
                exponets.push_back(exp);
                myfile >> coef;
                coefs_s.push_back(coef);
                myfile >> coef;
                coefs_p.push_back(coef);
            }
            return_vec.push_back(exponets);
            return_vec.push_back(coefs_s);
            return_vec.push_back(coefs_p);
        }
    }

    basis_map[atomic_num] = return_vec;
}

// in order -B, 0.5(I_s + A_s), 0.5(I_p + A_p)
// params_map[1] = {9, 7.176, 0};
// params_map[6] = {21, 14.051, 5.572};
// params_map[7] = {25, 19.316, 7.275};
// params_map[8] = {31, 25.390, 9.111};
// params_map[9] = {39, 32.272, 11.080};

void CNDO2::calc_basis()
{

    for (int i = 0; i < atoms.size(); i++)
    {

        vector<double> norms;
        vector<int> orb = {0, 0, 0};
        ////

        int atom_num = atoms[i].first;
        Function fun = Function(atom_num, atoms[i].second, norms, basis_map[atom_num][0], basis_map[atom_num][1], orb, params_map[atom_num][0], params_map[atom_num][1], electron_map[atom_num], 0, 0, 0, 1);
        basis.push_back(fun);
        s_basis.push_back(fun);
        int check = 0;
        if (atom_num == 1 && check == 0)
        {
            vector<Function> hfun = {fun};
            A_basis.push_back(hfun);
            check++;
        }
        else
        {
            vector<Function> nonfun;
            if (atom_num != 1)
            {
                int count = 0;
                for (int m = -1; m < 2; m++)
                {
                    vector<double> norms2;
                    vector<int> orb2;

                    if (count == 0)
                    {
                        orb2 = {1, 0, 0};
                    }
                    else if (count == 1)
                    {
                        orb2 = {0, 1, 0};
                    }

                    else if (count == 2)
                    {
                        orb2 = {0, 0, 1};
                    }

                    Function fun2 = Function(atom_num, atoms[i].second, norms2, basis_map[atom_num][0], basis_map[atom_num][2], orb2, params_map[atom_num][0], params_map[atom_num][2], electron_map[atom_num], 0, m, 0, 0);
                    nonfun.push_back(fun2);
                }
            }
            else
            {
                nonfun.push_back(fun);
            }
        }

        if (atom_num != 1)
        {
            int count = 0;
            for (int m = -1; m < 2; m++)
            {
                vector<double> norms2;
                vector<int> orb2;

                if (count == 0)
                {
                    orb2 = {1, 0, 0};
                }
                else if (count == 1)
                {
                    orb2 = {0, 1, 0};
                }

                else if (count == 2)
                {
                    orb2 = {0, 0, 1};
                }

                Function fun2 = Function(atom_num, atoms[i].second, norms2, basis_map[atom_num][0], basis_map[atom_num][2], orb2, params_map[atom_num][0], params_map[atom_num][2], electron_map[atom_num], 0, m, 0, 0);

                basis.push_back(fun2);

                count++;
            }
        }
    }
}

double CNDO2::p_uv(arma::Col<double> C_U, arma::Col<double> C_V, electrons spin)
{
    int n;
    arma::Mat<double> temp_C(N, N);

    if (spin == alpha_)
    {
        n = alpha;
    }
    else if (spin == beta_)
    {
        n = beta;
    }

    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += C_U[i] * C_V[i];
    }
    return sum;
}

/// @param C_U should be a row u from the C matrix
/// @param C_V should be a row v from the C matrix
double CNDO2::p_uv_tot(int U, int V)
{
    return p_uv(C_alpha.col(U), C_alpha.col(V), alpha_) + p_uv(C_beta.col(U), C_beta.col(V), beta_);
}

void CNDO2::add_norms()
{
    for (int k = 0; k < basis.size(); k++)
    {
        Function &fun = basis[k];
        for (int i = 0; i < 3; i++)
        {
            double overlap = 1;

            for (int j = 0; j < 3; j++)
            {
                overlap *= overlap_integral_1d_analytical(fun.R[j], fun.R[j], fun.orbital[j], fun.orbital[j], fun.exponets[i], fun.exponets[i]);
            }
            fun.norms.push_back(1 / pow(overlap, 0.5));
        }
    }
}

double CNDO2::P_tot(Function &fun)
{
    double sum = 0;
    int u = 0;
    // if (spin == alpha_)
    //{
    for (auto &current_fun : basis)
    {
        // this only works for molecules that are two atom only a
        if (current_fun.R == fun.R)
        {
            sum += p_uv_tot(u, u);
            // sum += P
        }
        u++;
    }
    //}
    // else
    // {
    //     for (auto &fun : basis)
    //     {
    //         if (fun.atomic_num = atomic_num)
    //         {
    //             sum += p_uv_tot(C_beta.col(u), C_beta.col(u));
    //         }
    //         u++;
    //     }
    //}//

    return sum;
}

// will always ve 2x2 for our style problem

// want gamma_XX gamma_XY
//  gamma_XY gamma_YY

// these shoud be Aorbitial of atom_A and Sorbitial

double CNDO2::gamma_AB(Function &funA, Function &funB)
{
    // double RA_RB_squared = pow(funA.R[0] - funB.R[0], 2) + pow(funA.R[1] - funB.R[1], 2) + pow(funA.R[2] - funB.R[2], 2);
    double zero_0;

    double sum = 0;
    for (int k = 0; k < 3; k++)
    {
        for (int kk = 0; kk < 3; kk++)
        {
            for (int l = 0; l < 3; l++)
            {
                for (int ll = 0; ll < 3; ll++)
                {
                    double sigma_a = 1 / (funA.exponets[k] + funA.exponets[kk]);
                    double sigma_b = 1 / (funB.exponets[l] + funB.exponets[ll]);
                    double U_A = pow(M_PI * sigma_a, 1.5);
                    double U_B = pow(M_PI * sigma_b, 1.5);
                    double V_2 = 1 / (sigma_a + sigma_b);
                    double dist = sqrt(pow((funA.R[0] - funB.R[0]), 2) +
                                       pow((funA.R[1] - funB.R[1]), 2) +
                                       pow((funA.R[2] - funB.R[2]), 2));

                    double T = V_2 * pow(dist, 2);
                    zero_0 = U_A * U_B * sqrt(1 / pow(dist, 2)) * erf(sqrt(T));

                    sum += (funA.coefs[k] * funA.norms[k]) * (funA.coefs[kk] * funA.norms[kk]) * (funB.coefs[l] * funB.norms[l]) * (funB.coefs[ll] * funB.norms[ll]) * zero_0;
                }
            }
        }
    }
    return sum * 27.211;
}

void CNDO2::get_s_basis()
{

    for (auto &fun : basis)
    {
        if (fun.orbital == vector<int>{0, 0, 0})
        {
            // Function temp_f = fun;
            s_basis.push_back(fun);
        }
    }
}

double CNDO2::gamma_AA(Function &funA, Function &funB)
{
    // double RA_RB_squared = pow(funA.R[0] - funB.R[0], 2) + pow(funA.R[1] - funB.R[1], 2) + pow(funA.R[2] - funB.R[2], 2);
    double zero_0;

    double sum = 0;
    for (int k = 0; k < 3; k++)
    {
        for (int kk = 0; kk < 3; kk++)
        {
            for (int l = 0; l < 3; l++)
            {
                for (int ll = 0; ll < 3; ll++)
                {
                    double sigma_a = 1 / (funA.exponets[k] + funA.exponets[kk]);
                    double sigma_b = 1 / (funB.exponets[l] + funB.exponets[ll]);
                    double U_A = pow(M_PI * sigma_a, 1.5);
                    double U_B = pow(M_PI * sigma_b, 1.5);
                    double V_2 = 1 / (sigma_a + sigma_b);
                    // double T = V_2 * RA_RB_squared;qu
                    zero_0 = U_A * U_B * sqrt(2 * V_2) * sqrt(2 / M_PI);

                    sum += (funA.coefs[k] * funA.norms[k]) * (funA.coefs[kk] * funA.norms[kk]) * (funB.coefs[l] * funB.norms[l]) * (funB.coefs[ll] * funB.norms[ll]) * zero_0;
                }
            }
        }
    }
    return sum * 27.211;
}

// double CNDO2::gamma_AB(Function &funA, Function &funB)
// {
//     // std::function<double()> zero_0;
//     double zero_0;
//     double sum = 0;
//     double RA_RB_squared = pow(funA.R[0] - funB.R[0], 2) + pow(funA.R[1] - funB.R[1], 2) + pow(funA.R[2] - funB.R[2], 2);

//     for (int k = 0; k < 3; k++)
//     {
//         for (int k_ = 0; k_ < 3; k_++)
//         {
//             for (int l = 0; l < 3; l++)
//             {
//                 for (int l_ = 0; l_ < 3; l_++)
//                 {

//                     if (funA.R != funB.R)
//                     {

//                         // zero_0 = [&funA, &funB, k, k_, l, l_]()

//                         // {

//                         cout
//                             << funA.atomic_num << endl;
//                         for (int i = 0; i < 3; i++)
//                         {
//                             cout << funA.exponets[k] << endl;
//                         }
//                         double sigma_a = 1 / (funA.exponets[k] + funA.exponets[k_]);
//                         double sigma_b = 1 / (funB.exponets[l] + funB.exponets[l_]);
//                         double V_2 = 1 / (sigma_a + sigma_b);

//                         double U_A = pow(M_PI * sigma_a, 3 / 2);
//                         double U_B = pow(M_PI * sigma_b, 3 / 2);
//                         double U = U_A * U_B;

//                         // double RA_RB = pow(pow(funA.R[0] - funB.R[0], 2) + pow(funA.R[1] - funB.R[1], 2) + pow(funA.R[2] - funB.R[2], 2), 0.5);

//                         // double T = V_2 * (pow(funA.R[0] - funB.R[0], 2) + pow(funA.R[1] - funB.R[1], 2) + pow(funA.R[2] - funB.R[2], 2));

//                         // double RA_RB_squared = pow(funA.R[0] - funB.R[0], 2) + pow(funA.R[1] - funB.R[1], 2) + pow(funA.R[2] - funB.R[2], 2);
//                         double T = V_2 * RA_RB_squared;

//                         zero_0 = (U * (1 / sqrt(RA_RB_squared)) * erf(sqrt(T)));
//                         // };
//                     }
//                     else
//                     {
//                         // zero_0 = [&funA, &funB, k, k_, l, l_]()
//                         //{
//                         double sigma_a = 1 / (funA.exponets[k] + funA.exponets[k_]);
//                         double sigma_b = 1 / (funB.exponets[l] + funB.exponets[l_]);
//                         double V_2 = 1 / (sigma_a + sigma_b);

//                         double U_A = pow(M_PI * sigma_a, 3 / 2);
//                         double U_B = pow(M_PI * sigma_b, 3 / 2);
//                         double U = U_A * U_B;

//                         double RA_RB_squared = pow(funA.R[0] - funB.R[0], 2) + pow(funA.R[1] - funB.R[1], 2) + pow(funA.R[2] - funB.R[2], 2);
//                         double T = V_2 * RA_RB_squared;

//                         zero_0 = U * pow(2 * V_2, 0.5) * sqrt(2 / M_PI);
//                         //};
//                     }
//                     sum += (funA.coefs[k] * funA.norms[k]) * (funA.coefs[k_] * funA.norms[k_]) * (funB.coefs[l] * funB.norms[l]) * (funB.coefs[l_] * funB.norms[l_]) * zero_0;
//                 }
//             }
//         }
//     }
//     return sum * 27.211;
// }

// double CNDO2::gamma_AB(){

// }

void CNDO2::make_gamma()
{
    // vector<Function> s_basis;

    // Function & funB;

    // for (Function &fun : basis)
    // {
    //     if (fun.orbital == vector<int>{0, 0, 0})
    //     {
    //         // s_basis.push_back(fun);
    //         s_basis.push_back(fun);
    //     }
    //     // else if (fun.orbital == vector<int>{0, 0, 0} && fun.atomic_num == atom_B)
    //     // {
    //     //     s_basis[1] = fun;
    //     // }
    //     // else
    //     // {
    //     //     continue;
    // }

    // cout << "!!!!!!" << end1;

    // G.at(0, 0) = gamma_AB(basis[0], basis[0]);

    // G.at(0, 1) = gamma_AB(basis[0], basis[1]);
    // G.at(1, 0) = gamma_AB(basis[0], basis[1]);
    // G.at(1, 1) = gamma_AB(basis[1], basis[1]);

    // xx xyx
    // yx yy
    cout << "basis size: " << s_basis.size();
    cout << "N: " << this->N << endl;

    int c1;
    int c2;

    c1 = 0;
    for (auto &fun : basis)
    {
        if (fun.orbital == vector<int>{0, 0, 0})
        {
            // arma::Col<double> temp_col;
            c2 = 0;
            for (auto &fun2 : basis)
            {
                if (c1 == c2)
                {
                    G.at(c1, c2) = gamma_AA(fun, fun);
                }
                else
                {
                    G.at(c1, c2) = gamma_AB(fun, fun2);
                }
                c2++;
            }
            // G.col(c1) = temp_col;
        }
        c1++;
    }

    // for (int i = 0; i < s_basis.size(); i++)
    // {

    //     for (int j = 0; j < s_basis.size(); j++)
    //     {
    //         if (i != j)
    //         {

    //             // cout << "breaking ..." << endl;
    //             // break;
    //             G.at(i, j) = gamma_AB(s_basis[i], s_basis[j]);
    //         }
    //         else
    //         {
    //             G.at(i, j) = gamma_AA(s_basis[i], s_basis[j]);
    //         }
    //     }
    // }

    cout << "!!!" << endl;
    G.print();
}

void CNDO2::calc_P()
{

    // int n;
    // if (spin == alpha_)
    // {
    //     n = alpha;
    // }
    // else if (spin == beta_)
    // {
    //     n = beta;
    // }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum_alpha = 0;

            for (int k = 0; k < alpha; k++)
            {
                sum_alpha += C_alpha(i, k) * C_alpha(j, k);
            }
            P_alpha.at(i, j) = sum_alpha;

            double sum_beta = 0;
            for (int k = 0; k < beta; k++)
            {
                sum_alpha += C_beta(i, k) * C_beta(j, k);
            }
            P_beta.at(i, j) = sum_beta;
        }
    }
}

void CNDO2::update_G()
{
    arma::vec P_t = arma::zeros(num_atoms);
    // vector<Function> Aos;

    for (int k = 0; k < num_atoms; k++)
    {
        vector<Function> Aos;
        if (k == 0)

        {

            // vector<double> funA_center = funA.R;
            Aos.push_back(basis[0]);
            // vector<Function> Bos(basis.size() - 1);

            // for (int i = 0; i < basis.size() - 1; i++)
            // {
            //     Bos[i] = basis[i + 1];
            // }
        }
        else
        {

            // vector<double> funA_center = funA.R;

            // Aos.resize(basis.size() - 1);

            for (int i = 0; i < basis.size() - 1; i++)
            {
                Aos.push_back(basis[i + 1]);
            }

            // vector<Function> Bos = {basis[0]};
        }

        int kk = 0;
        for (auto Aao : Aos)
        {
            P_t(k) += P_alpha(kk, kk) + P_beta(kk, kk);
            kk++;
        }
    }
    P_t.print("P_t");

    int kk = 0;
    for (int k = 0; k < num_atoms; k++)
    {
        // if (k == 0)

        // {

        //     // vector<double> funA_center = funA.R;
        //     Aos.push_back(basis[0]);
        //     // vector<Function> Bos(basis.size() - 1);

        //     // for (int i = 0; i < basis.size() - 1; i++)
        //     // {
        //     //     Bos[i] = basis[i + 1];
        //     // }
        // }
        // else
        // {

        //     // vector<double> funA_center = funA.R;

        //     // Aos.resize(basis.size() - 1);

        //     for (int i = 0; i < basis.size() - 1; i++)
        //     {
        //         Aos.push_back(basis[i + 1]);
        //     }

        //     // vector<Function> Bos = {basis[0]};
        // }

        // int kk = 0;
        // for (auto Aao : Aos)
        // {
        //     P_t(k) += P_alpha(kk, kk) + P_beta(kk, kk);
        //     kk++;
        // }
        vector<Function> Aos;

        if (k == 0)

        {

            // vector<double> funA_center = funA.R;
            Aos.push_back(basis[0]);
            // vector<Function> Bos(basis.size() - 1);

            // for (int i = 0; i < basis.size() - 1; i++)
            // {
            //     Bos[i] = basis[i + 1];
            // }
        }
        else
        {

            // vector<double> funA_center = funA.R;

            // Aos.resize(basis.size() - 1);

            for (int i = 0; i < basis.size() - 1; i++)
            {
                Aos.push_back(basis[i + 1]);
            }

            // vector<Function> Bos = {basis[0]};
        }

        double gammaAA = G(k, k);

        for (auto Aao : Aos)
        {
            Ga(kk, kk) = (P_t(k) - P_alpha(kk, kk)) * gammaAA;
            Gb(kk, kk) = (P_t(k) - P_beta(kk, kk)) * gammaAA;
            int jj = 0;
            for (int j = 0; j < num_atoms; j++)
            {

                double gammaAB = G(k, j);
                if (k != j)
                {
                    Ga(kk, kk) += P_t(j) * gammaAB;
                    Gb(kk, kk) += P_t(j) * gammaAB;
                }
                vector<Function> Bos;

                if (j == 0)

                {
                    // vector<double> funA_center = funA.R;
                    Bos.push_back(basis[0]);
                    // vector<Function> Bos(basis.size() - 1);

                    // for (int i = 0; i < basis.size() - 1; i++)
                    // {
                    //     Bos[i] = basis[i + 1];
                    // }
                }
                else
                {
                    // funA = basis[1];
                    // vector<double> funA_center = funA.R;

                    // Bos.resize(basis.size() - 1);

                    for (int i = 0; i < basis.size() - 1; i++)
                    {
                        Bos.push_back(basis[i + 1]);
                    }

                    // vector<Function> Bos = {basis[0]};
                }
                for (auto Bao : Bos)
                {
                    if (kk != j)
                    {
                        Ga(kk, jj) = -gammaAB * P_alpha(kk, jj);
                        Gb(kk, jj) = -gammaAB * P_beta(kk, jj);
                        // cout << k_AO << "  " << j_AO<< "  " << gammaAB<<"  " << Pa(k_AO, j_AO)<< "  " << Ga(k_AO, j_AO)<< endl;
                    }
                    jj++;
                }
            }
            kk++;
        }
    }
    Ga.print("Ga");
    Gb.print("Gb");
}

void CNDO2::calc_f()
{

    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         if (i != j)
    //         {
    //             // f.at(i, j) = 0.5 * (basis[i].B + basis[j].B) * S(i, j) - P(i, j) * gamma_AB(basis[i], basis[j]);
    //             f.at(i, j) = -0.5 * (basis[i].B + basis[j].B) * S(i, j) - P(i, j) * G(0, 1);
    //         }
    //         else
    //         {

    //             // double sum = 0;
    //             // for (auto &fun : basis)
    //             // {
    //             //     if (fun.R != basis[i].R)
    //             //     {
    //             //         sum += (P_tot(fun) - fun.val_elec) * G(0, 1);
    //             //         // sum += (P_tot(fun) - fun.val_elec) * G() gamma_AB(basis[i], fun);
    //             //     }
    //             // }

    //             f.at(i, i) = (-1 * (basis[i].I_A) + ((P_tot(basis[i]) - basis[i].val_elec) - (P(i, j) - 0.5)) * gamma_AA(basis[i], basis[i]));

    //             double sum = 0;
    //             for (auto &fun : basis)
    //             {
    //                 int count1 = 0;
    //                 int count2 = 0;
    //                 //&& fun.orbitial == vector<int>{0,0,0}
    //                 // fun.orbital == vector<int>{0, 0, 0} &&
    //                 //
    //                 if (fun.R[0] != basis[i].R[0] || fun.R[1] != basis[i].R[1] || fun.R[2] != basis[i].R[2])
    //                 {
    //                     // sub += fun.val_elec * gamma_AB(basis[i], fun);
    //                     sum += (P_tot(fun) - fun.val_elec) * G(0, 1);
    //                     break;
    //                 }
    //             }

    //             f.at(i, i) += sum;
    //         }

    //         // should maybe write a vector to keep each atom in individually and then can expand to its basis
    //         //  i think i might have actually done it this way on assignment 3
    //         // for (int i = 0; i < num_atoms_in_total; i++)
    //         //     if (A != B)
    //         //     {
    //         //         f.at(i, i) += (P_BB_total - valance_atom_num_B) * gamma_AB;
    //         //         // here B is any but the current atom we are on
    //         //         // since this loop is per basis function we need to make sure somehow that this is handled appropiatly

    //         //         // if element number identifier, as in we can call what atom it is based on the index
    //         //     }
    //     }
    // }
    update_G();
    F_alpha = core_H + Ga;
    F_beta = core_H + Gb;
}

void CNDO2::calc_C_and_e()
{

    // arma::Col<double> eigval;
    // arma::Mat<double> eigvec;
    eig_sym(e_alpha, C_alpha, F_alpha);
    eig_sym(e_beta, C_beta, F_beta);

    // C = eigvec;
}

void CNDO2::make_map()
{
    read_basis("H_STO3G.txt", 1);

    read_basis("C_STO3G.txt", 6);

    read_basis("N_STO3G.txt", 7);

    read_basis("O_STO3G.txt", 8);

    read_basis("F_STO3G.txt", 9);

    // keeps in order but not neccarly, might be good for the first entery tho so we know first map is H.
}

void CNDO2::calc_S()
{
    cout << N << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {

            S.at(i, j) = 0.0;
            Function &u = basis[i];
            Function &v = basis[j];
            cout << "u[i =" << i << "] exp size" << u.exponets.size() << endl;
            cout << "v [j =" << j << "] exp size" << v.exponets.size() << endl;

            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    double overlap = 1;

                    for (int z = 0; z < 3; z++)
                    {
                        overlap *= overlap_integral_1d_analytical(u.R[z], v.R[z], u.orbital[z], v.orbital[z], u.exponets[k], v.exponets[l]);
                    }

                    S.at(i, j) += overlap * u.norms[k] * v.norms[l] * u.coefs[k] * v.coefs[l];
                }
            }
        }
    }
}

int CNDO2::factorial(int num)
{
    int product = 1;
    if (num == -1)
    {
    }
    else
    {
        for (int i = 1; i < num + 1; i++)
        {
            product *= i;
        }
    }

    return product;
}

int CNDO2::double_factorial(int num)
{
    int product = 1;
    if (num % 2 == 0)
    {
        for (int i = 2; i < num + 1; i += 2)
        {
            product *= i;
        }
    }
    else
    {
        for (int i = 1; i < num + 1; i += 2)
        {
            product *= i;
        }
    }
    return product;
}

double CNDO2::X_p(double X_A, double X_B, double aalpha, double bbeta)
{
    return (aalpha * X_A + bbeta * X_B) / (aalpha + bbeta);
}

int CNDO2::binomial_coefficient(int m, int n)
{
    return factorial(m) / (factorial(n) * factorial(m - n));
}

double CNDO2::overlap_integral_1d_analytical(double X_A, double X_B, int l_A, int l_B, double aalpha, double bbeta)
{
    double sum = 0;
    double X_P = (aalpha * X_A + bbeta * X_B) / (aalpha + bbeta);
    for (int i = 0; i < l_A + 1; i++)
    {
        for (int j = 0; j < l_B + 1; j++)
        {
            if ((i + j) % 2 == 0)
            {
                sum += binomial_coefficient(l_A, i) * binomial_coefficient(l_B, j) * double_factorial(i + j - 1) * pow(X_P - X_A, l_A - i) * pow(X_P - X_B, l_B - j) / pow(2 * (aalpha + bbeta), (i + j) / 2);
            }
            else
            {
                continue;
            }
        }
    }
    double result = exp(-aalpha * bbeta * pow(X_A - X_B, 2) / (aalpha + bbeta)) * pow(M_PI / (aalpha + bbeta), 0.5) * sum;
    return result;
}

double CNDO2::inf_norm(arma::Mat<double> &P_new, arma::Mat<double> &P_old)
{

    double max_diff = 0;
    for (int i = 0; i < P_new.n_rows; i++)
    {
        for (int j = 0; j < P_new.n_rows; j++)
        {
            double diff = abs(P_new(i, j) - P_old(i, j));
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

void CNDO2::huckle(arma::Mat<double> &H)
{

    int kk = 0;
    for (int k = 0; k < num_atoms; k++)
    {

        // will only wokr if H is first atom and we only have two atoms in total
        Function funA = basis[k];
        vector<Function> Aos;
        if (k == 0)

        {

            vector<double> funA_center = funA.R;
            Aos.push_back(basis[0]);
            // vector<Function> Bos(basis.size() - 1);

            // for (int i = 0; i < basis.size() - 1; i++)
            // {
            //     Bos[i] = basis[i + 1];
            // }
        }
        else
        {

            vector<double> funA_center = funA.R;

            // Aos.resize(basis.size() - 1);

            for (int i = 0; i < basis.size() - 1; i++)
            {
                Aos.push_back(basis[i + 1]);
            }

            // vector<Function> Bos = {basis[0]};
        }

        double ZA = funA.val_elec;
        double A_IA = funA.I_A;
        double betaA = funA.B;

        for (auto &fun_a : Aos)
        {
            H.at(kk, kk) = ((-1) * fun_a.I_A) - (ZA - 0.5) * gamma_AA(fun_a, fun_a);

            int jj = 0;

            for (int j = 0; j < num_atoms; j++)
            {

                Function &funB = basis[j];
                vector<double> funB_center = funB.R;
                vector<Function> Bos;

                if (j == 0)

                {
                    // vector<double> funA_center = funA.R;
                    Bos.push_back(basis[0]);
                    // vector<Function> Bos(basis.size() - 1);

                    // for (int i = 0; i < basis.size() - 1; i++)
                    // {
                    //     Bos[i] = basis[i + 1];
                    // }
                }
                else
                {
                    // funA = basis[1];
                    // vector<double> funA_center = funA.R;

                    // Bos.resize(basis.size() - 1);

                    for (int i = 0; i < basis.size() - 1; i++)
                    {
                        Bos.push_back(basis[i + 1]);
                    }

                    // vector<Function> Bos = {basis[0]};
                }
                double ZB = funB.val_elec;
                cout << "ZB: " << ZB << endl;

                double B_IA = funB.I_A;
                double betaB = funB.B;

                double avg_beta = -(betaA + betaB) / 2;

                if (k != j)

                    H.at(kk, kk) = H(kk, kk) - double(ZB) * gamma_AB(basis[k], basis[j]);

                for (auto &fun_b : Bos)
                {
                    if (kk != jj)

                        H.at(kk, jj) = avg_beta * S(kk, jj);

                    jj++;
                }
            }
            kk++;
        }
    }
}

//         for (int j = 0; j < num_atoms; j++)
//         {
//         }
//     }

//     for (int i = 0; i < N; i++)
//     {
//         for (int j = 0; j < N; j++)
//         {
//             if (i != j)
//             {
//                 // should this be neg??
//                 H.at(i, j) = 0.5 * ((-1 * basis[i].B) + (-1 * basis[j].B)) * S(i, j);
//             }
//             else
//             {
//                 // int b_num;
//                 // for (auto &atom1: atoms)
//                 // {
//                 //     if (atom1.first != basis[i].atomic_num)
//                 //     {
//                 //         b_num = atom.first;
//                 //         break;
//                 //     }
//                 // }

//                 // if (basis[i].atomic_num == 1)
//                 // {

//                 //     cout << "atom_A =" << atom_A << endl;
//                 //     H.at(i, i) = ((-1) * basis[i].I_A) - ((basis[i].val_elec - 0.5) * G(0, 0));
//                 // }
//                 // else
//                 // {
//                 //     H.at(i, i) = ((-1) * basis[i].I_A) - ((basis[i].val_elec - 0.5) * G(1, 1));
//                 // }
//                 H.at(i, i) = ((-1) * basis[i].I_A) - ((basis[i].val_elec - 0.5) * G(0, 0));
//                 //(electron_map[atoms[1].first] * G(0, 1));

//                 double sub = 0;
//                 for (auto &fun : basis)
//                 {
//                     int count1 = 0;
//                     int count2 = 0;
//                     //&& fun.orbitial == vector<int>{0,0,0}
//                     // fun.orbital == vector<int>{0, 0, 0} &&
//                     //
//                     if (fun.R[0] != basis[i].R[0] || fun.R[1] != basis[i].R[1] || fun.R[2] != basis[i].R[2])
//                     {
//                         // sub += fun.val_elec * gamma_AB(basis[i], fun);
//                         sub += fun.val_elec * G(0, 1);
//                         // break;
//                     }
//                 }
//                 H.at(i, i) -= sub;
//             }
//         }
//     }
// }

void CNDO2::make_Xuv()
{

    for (int u = 0; u < basis.size(); u++)
    {
        for (int v = 0; v < basis.size(); v++)
        {
            X.at(u, v) = (basis[u].B + basis[v].B) * (P_alpha(u, v) + P_beta(u, v));
        }
    }
}

double CNDO2::yAB()
{

    vector<Function> atom_A = A_basis[0];
    vector<Function> atom_B = A_basis[1];

    double p_AA = 0;
    double p_BB = 0;

    for (int uu = 0; uu < basis.size(); uu++)
    {
        if (basis[uu].R[0] == atom_A[0].R[0] && basis[uu].R[1] == atom_A[0].R[1] && basis[uu].R[2] == atom_A[0].R[2])
        {
            p_AA += (P_alpha(uu, uu) + P_beta(uu, uu));
        }
        else
        {
            p_BB += (P_alpha(uu, uu) + P_beta(uu, uu));
        }
    }

    double y_AB = p_AA * p_BB - atom_B[0].val_elec * p_AA - atom_A[0].val_elec * p_BB;
    double sum = 0;
    for (int u = 0; basis.size(); u++)
    {
        if (basis[u].R[0] == atom_A[0].R[0] && basis[u].R[1] == atom_A[0].R[1] && basis[u].R[2] == atom_A[0].R[2])
        {
            for (int v = 0; v < basis.size(); v++)
            {
                if (basis[v].R[0] == atom_B[0].R[0] && basis[v].R[1] == atom_B[0].R[1] && basis[v].R[2] == atom_B[0].R[2])
                {
                    sum += (P_alpha(u, v) * P_alpha(u, v) + P_beta(u, v) * P_beta(u, v));
                }
                else
                {
                    continue;
                }
            }
        }
        else
        {
            continue;
        }
    }

    y_AB -= sum;
    return y_AB;
}

double CNDO2::yBA()
{

    vector<Function> atom_A = A_basis[1];
    vector<Function> atom_B = A_basis[0];

    double p_AA = 0;
    double p_BB = 0;

    for (int uu = 0; uu < basis.size(); uu++)
    {
        if (basis[uu].R[0] == atom_A[0].R[0] && basis[uu].R[1] == atom_A[0].R[1] && basis[uu].R[2] == atom_A[0].R[2])
        {
            p_AA += (P_alpha(uu, uu) + P_beta(uu, uu));
        }
        else
        {
            p_BB += (P_alpha(uu, uu) + P_beta(uu, uu));
        }
    }

    double y_AB = p_AA * p_BB - atom_B[0].val_elec * p_AA - atom_A[0].val_elec * p_BB;
    double sum = 0;
    for (int u = 0; basis.size(); u++)
    {
        if (basis[u].R[0] == atom_A[0].R[0] && basis[u].R[1] == atom_A[0].R[1] && basis[u].R[2] == atom_A[0].R[2])
        {
            for (int v = 0; v < basis.size(); v++)
            {
                if (basis[v].R[0] == atom_B[0].R[0] && basis[v].R[1] == atom_B[0].R[1] && basis[v].R[2] == atom_B[0].R[2])
                {
                    sum += (P_alpha(u, v) * P_alpha(u, v) + P_beta(u, v) * P_beta(u, v));
                }
                else
                {
                    continue;
                }
            }
        }
        else
        {
            continue;
        }
    }

    y_AB -= sum;
    return y_AB;
}

void CNDO2::make_grad()
{
    arma::Mat<double> g_a;

    double sum = 0;
    for (int u = 0; basis.size(); u++)
    {

        for (int v = 0; v < basis.size(); v++)
        {

            if (basis[u].R[0] != basis[v].R[0] || basis[u].R[1] != basis[v].R[1] || basis[u].R[2] != basis[v].R[2])
            {
                g_a.at(u, v) =
            }
        }
    }
}

double CNDO2::total_energy()
{
    double sum = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            sum += P_alpha(i, j) * (core_H(i, j) + F_alpha(i, j)) + P_beta(i, j) * (core_H(i, j) + F_beta(i, j));
        }
    }
    double dist = pow(pow(atoms[0].second[0] + atoms[1].second[0], 2) + pow(atoms[0].second[1] + atoms[1].second[1], 2) + pow(atoms[0].second[2] + atoms[1].second[2], 2), 0.5);
    sum = 0.5 * sum + ((electron_map[atom_A] * electron_map[atom_B]) / dist);

    return sum;
}

// double CNDO2::inf_norm(arma::Mat<double> &P_new, arma::Mat<double> &P_old)
// {

//     double max_diff = 0;
//     for (int i = 0; i < P_new.n_rows; i++)
//     {
//         for (int j = 0; j < P_new.n_rows; j++)
//         {
//             double diff = abs(P_new(i, j) - P_old(i, j));
//             if (diff > max_diff)
//             {
//                 max_diff = diff;
//             }
//         }
//     }
//     return max_diff;
// }

void CNDO2 ::Suv_RA()
{

    arma::Mat<double> S_ra;
    arma::Mat<double> S_rx;
    arma::Mat<double> S_ry;
    arma::Mat<double> S_rz;

    cout << "num cols: " << S_ra.n_cols << endl;
    cout << "num rows: " << S_ra.n_rows << endl;

    // for (int i = 0; i < basis.size(); i++)
    // {
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {

            // using (j,i) indexing since im constructing as matrix that i will then flatten;
            // armadillo flattens by rows so i need to create my independent partial matrixs as transposes
            S_rx.at(i, j) = 0.0;
            S_ry.at(i, j) = 0.0;
            S_rz.at(i, j) = 0.0;
            Function &u = basis[i];
            Function &v = basis[j];
            double pIx = 0;
            double pIy = 0;
            double pIz = 0;
            double Ix = 0;
            double Iy = 0;
            double Iz = 0;

            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {

                    pIx = -1 * u.orbital[0] * overlap_integral_1d_analytical(u.R[0], v.R[0], u.orbital[0] - 1, v.orbital[0], u.exponets[k], v.exponets[l]) +
                          2 * u.exponets[k] * overlap_integral_1d_analytical(u.R[0], v.R[0], u.orbital[0] + 1, v.orbital[0], u.exponets[k], v.exponets[l]);
                    pIy = -1 * u.orbital[1] * overlap_integral_1d_analytical(u.R[1], v.R[1], u.orbital[1] - 1, v.orbital[1], u.exponets[k], v.exponets[l]) +
                          2 * u.exponets[k] * overlap_integral_1d_analytical(u.R[1], v.R[1], u.orbital[1] + 1, v.orbital[1], u.exponets[k], v.exponets[l]);
                    pIz = -1 * u.orbital[2] * overlap_integral_1d_analytical(u.R[2], v.R[2], u.orbital[2] - 1, v.orbital[2], u.exponets[k], v.exponets[l]) +
                          2 * u.exponets[k] * overlap_integral_1d_analytical(u.R[2], v.R[2], u.orbital[2] + 1, v.orbital[2], u.exponets[k], v.exponets[l]);
                    Ix = overlap_integral_1d_analytical(u.R[0], v.R[0], u.orbital[0], v.orbital[0], u.exponets[k], v.exponets[l]);
                    Iy = overlap_integral_1d_analytical(u.R[1], v.R[1], u.orbital[1], v.orbital[1], u.exponets[k], v.exponets[l]);
                    Iz = overlap_integral_1d_analytical(u.R[2], v.R[2], u.orbital[2], v.orbital[2], u.exponets[k], v.exponets[l]);
                    S_rx.at(i, j) += u.norms[k] * v.norms[l] * u.coefs[k] * v.coefs[l] * pIx * Iy * Iz;
                    S_ry.at(i, j) += u.norms[k] * v.norms[l] * u.coefs[k] * v.coefs[l] * Ix * pIy * Iz;
                    S_rz.at(i, j) += u.norms[k] * v.norms[l] * u.coefs[k] * v.coefs[l] * Ix * Iy * pIz;

                    if (u.R[0] == v.R[0] && u.R[1] == v.R[1] && u.R[2] == v.R[2])
                    {
                        S_rx.at(i, j) = 0;
                        S_ry.at(i, j) = 0;
                        S_rz.at(i, j) = 0;
                    }
                }
            }
        }
    }

    arma::rowvec S_rx_flat = vectorise(S_rx.t()).t();
    arma::rowvec S_ry_flat = vectorise(S_ry.t()).t();
    arma::rowvec S_rz_flat = vectorise(S_rz.t()).t();

    S_ra.row(0) = S_rx_flat;
    S_ra.row(1) = S_ry_flat;
    S_ra.row(2) = S_rz_flat;

    // S_rx.print();
    // S_ry.print();
    // S_rz.print();

    S_ra.print();
}

void CNDO2 ::gamma_RA()
{

    arma::Mat<double> gamma_ra;
    arma::Mat<double> gamma_rx;
    arma::Mat<double> gamma_ry;
    arma::Mat<double> gamma_rz;

    gamma_ra.set_size(3, pow(N, 2));
    gamma_rx.set_size(N, N);
    gamma_ry.set_size(N, N);
    gamma_rz.set_size(N, N);
    double sum = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            Function &funA = basis[i];
            Function &funB = basis[j];
            double zero_0dx = 0;
            double zero_0dy = 0;
            double zero_0dz = 0;
            gamma_rx.at(i, j) = 0;
            gamma_ry.at(i, j) = 0;
            gamma_rz.at(i, j) = 0;

            for (int k = 0; k < 3; k++)
            {
                for (int kk = 0; kk < 3; kk++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        for (int ll = 0; ll < 3; ll++)
                        {
                            Function &funC = funA;

                            double sigma_a = 1 / (funA.exponets[k] + funA.exponets[kk]);
                            double sigma_b = 1 / (funB.exponets[l] + funB.exponets[ll]);
                            double U_A = pow(M_PI * sigma_a, 1.5);
                            double U_B = pow(M_PI * sigma_b, 1.5);
                            double V_2 = 1 / (sigma_a + sigma_b);
                            double dist = sqrt(pow((funA.R[0] - funB.R[0]), 2) +
                                               pow((funA.R[1] - funB.R[1]), 2) +
                                               pow((funA.R[2] - funB.R[2]), 2));

                            double T = V_2 * pow(dist, 2);

                            double constants = (funA.coefs[k] * funA.norms[k]) * (funA.coefs[kk] * funA.norms[kk]) * (funB.coefs[l] * funB.norms[l]) * (funB.coefs[ll] * funB.norms[ll]);

                            // auto satus = [&]()
                            // {
                            //     bool is_true = true;
                            //     if (((funA.R[0] == funC.R[0] && funA.R[1] == funC.R[1] && funA.R[2] == funC.R[2]) && (funC.R[0] != funB.R[0] || funC.R[1] != funB.R[1] ||
                            //                                                                                           funA.R[2] == funC.R[2] && funC.R[0] != funB.R[2])) ||
                            //         ((funB.R[0] == funC.R[0] && funB.R[1] == funC.R[1] && funB.R[2] == funC.R[2]) &&
                            //          (funC.R[0] != funB.R[0] || funC.R[1] != funB.R[1] || funC.R[2] != funB.R[2])))
                            //     {
                            //         return is_true;
                            //     }
                            //     else
                            //     {
                            //         is_true = false;
                            //         return is_true;
                            //     }
                            // };

                            if (funA.R[0] == funB.R[0] && funA.R[1] == funB.R[1] && funA.R[2] == funB.R[2])
                            {
                                gamma_rx.at(i, j) = 0;
                                gamma_ry.at(i, j) = 0;
                                gamma_rz.at(i, j) = 0;
                            }
                            else
                            {
                                zero_0dx = ((U_A * U_B * (funA.R[0] - funB.R[0])) / pow(dist, 2)) * ((-1 * (erf(sqrt(T)) / dist)) + ((2 * sqrt(V_2) / sqrt(M_PI)) * exp(-T)));
                                zero_0dy = ((U_A * U_B * (funA.R[1] - funB.R[1])) / pow(dist, 2)) * ((-1 * (erf(sqrt(T)) / dist)) + ((2 * sqrt(V_2) / sqrt(M_PI)) * exp(-T)));
                                zero_0dz = ((U_A * U_B * (funA.R[2] - funB.R[2])) / pow(dist, 2)) * ((-1 * (erf(sqrt(T)) / dist)) + ((2 * sqrt(V_2) / sqrt(M_PI)) * exp(-T)));

                                gamma_rx.at(i, j) += constants * zero_0dx;
                                gamma_ry.at(i, j) += constants * zero_0dy;
                                gamma_rz.at(i, j) += constants * zero_0dz;
                            }
                        }
                    }
                }
            }

            gamma_rx.at(i, j) *= 27.211;
            gamma_ry.at(i, j) *= 27.211;
            gamma_rz.at(i, j) *= 27.211;
        }
    }

    arma::rowvec gamma_rx_flat = vectorise(gamma_rx.t()).t();
    arma::rowvec gamma_ry_flat = vectorise(gamma_ry.t()).t();
    arma::rowvec gamma_rz_flat = vectorise(gamma_rz.t()).t();

    gamma_ra.row(0) = gamma_rx_flat;
    gamma_ra.row(1) = gamma_ry_flat;
    gamma_ra.row(2) = gamma_rz_flat;

    gamma_ra.print();
}

pair<double, double> CNDO2::one_step()
{
    calc_f();
    calc_f();

    calc_C_and_e();
    calc_C_and_e();

    arma::Mat<double> old_P_alpha = P_alpha;
    arma::Mat<double> old_P_beta = P_beta;
    calc_P();
    // calc_P());

    double max_alpha_change = inf_norm(old_P_alpha, P_alpha);
    double max_beta_change = inf_norm(old_P_beta, P_beta);

    // this->P_alpha = new_P_alpha;
    // this->P_beta = new_P_beta;
    return pair<double, double>(max_alpha_change, max_beta_change);
}

void CNDO2::info()
{

    cout << "gamma:" << endl;
    this->G.print();
    cout << endl;

    cout << "overlap:" << endl;
    S.print();
    cout << endl;

    cout << "core H:" << endl;
    core_H.print();
    cout << endl;
}

void CNDO2::run()
{

    cout << " Starting Guess P_alpha = P_beta = 0" << endl;
    cout << "P_alpha: " << endl;
    P_alpha.print();
    cout << "P_beta: " << endl;
    P_beta.print();

    int itt = 0;
    while (beta_change > 1e-6 || alpha_change > 1e-6)
    {
        cout << endl;
        cout << "iteration: " << itt << endl;
        // cout << "P_alpha: " << endl;
        // this->P_alpha.print();
        // cout << "P_beta: " << endl;
        // this->P_beta.print();

        cout << "F_alpha: " << endl;
        F_alpha.print();
        cout << "F_beta: " << endl;
        F_beta.print();

        pair<double, double> max_mag = one_step();

        cout << "after solving eigen equation: " << itt << endl;

        cout << "C_alpha: " << endl;
        C_alpha.print();
        cout << "C_beta: " << endl;
        C_beta.print();

        cout << "p = " << this->alpha << " and q = " << this->beta << endl;

        cout << "new P_alpha: " << endl;
        P_alpha.print();
        cout << "new P_beta: " << endl;
        P_beta.print();

        alpha_change = max_mag.first;
        beta_change = max_mag.second;
        // pair<double, double> max_mag = one_step();
        itt++;
    }

    cout << "Converged after " << itt << " iterations." << endl;
}

int main()
{

    // CNDO2 molecule = CNDO2("H2.txt");
    // molecule.info();
    // molecule.Suv_RA();
    // molecule.gamma_RA();

    // molecule.run();

    // double energy = molecule.total_energy();

    // cout << endl
    //      << "Total energy = " << energy << "." << endl;

    CNDO2 molecule2 = CNDO2("HF.txt");
    molecule2.info();
    molecule2.Suv_RA();
    molecule2.gamma_RA();

    molecule2.run();

    double energy2 = molecule2.total_energy();

    cout << endl
         << "Total energy = " << energy2 << "." << endl;

    return 0;
}