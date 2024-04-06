#include "hw4.h"

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

    // if( atom_A == 1 && atom_B ==1){
    //     N = 2*b
    // }
    // else if( atom_B ==1 && atom_A != 1){
    N = 4 * a + b;
    //}
    // else if ( atom_B != 1 && atom_A !=1){
    // N = W*
    //}

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

    // initiate C = zero matrix
    make_map();
    calc_basis();
    add_norms();

    calc_S();

    P.set_size(N, N);
    P.zeros();
    P_AA = P_tot(atom_A);
    P_BB = P_tot(atom_B);
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
        Function fun = Function(atom_num, atoms[i].second, norms, basis_map[atom_num][0], basis_map[atom_num][1], orb, params_map[atom_num][0], params_map[atom_num][1], electron_map[atom_num], 0, 0, 0, 0);
        basis.push_back(fun);
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
double CNDO2::p_uv_tot(arma::Col<double> C_U, arma::Col<double> C_V)
{
    return p_uv(C_U, C_V, alpha_) + p_uv(C_U, C_V, beta_);
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

double CNDO2::P_tot(int atomic_num)
{
    double sum = 0;
    int u = 0;
    for (auto &fun : basis)
    {
        if (fun.atomic_num = atomic_num)
        {
            sum += p_uv_tot(C.row(u), C.row(u));
        }
        u++;
    }

    return sum;
}

double CNDO2::gamma_AB(Function &funA, Function &funB)
{

    // double zero_0;
    //  Function & funA;
    //  Function & funB;

    // if (funA.R != funB.R)
    // {
    //     auto zero_0 = [&funA, &funB, k, k_, l, l_]()
    //     {
    //         double sigma_a = 1 / (funA.exponets[k] + funA.exponets[k_]);
    //         double sigma_b = 1 / (funB.exponets[l] + funB.exponets[l_]);
    //         double V_2 = 1 / (sigma_a + sigma_b);

    //         double U_A = pow(PI() * sigma_a, 3 / 2);
    //         double U_B = pow(PI() * sigma_b, 3 / 2);
    //         double U = U_A * U_B;

    //         double RA_RB_2 = pow(funA.R[0] + funB.R[0], 2) + pow(funA.R[1] + funB.R[1], 2) + pow(funB.R[2] + funB.R[2], 2);

    //         double T = V_2 * dist;

    //         return U * pow(dist, -1 / 2) * erf(T);
    //     }
    // }
    // else
    // {

    //     auto zero_0 = [&funA, &funb, k, k_, l, l_]()
    //     {
    //         double sigma_a = 1 / (funA.exponets[k] + funA.exponets[k_]);
    //         double sigma_b = 1 / (funB.exponets[l] + funB.exponets[l_]);
    //         double V_2 = 1 / (sigma_a + sigma_b);

    //         double U_A = pow(cmath.PI * sigma_a, 3 / 2);
    //         double U_B = pow(cmath.PI * sigma_b, 3 / 2);
    //         double U = U_A * U_B;

    //         return U * pow(2 * V_2, 1 / 2) * pow(2 / PI(), 1 / 2);
    //     }
    // }

    double sum = 0;
    for (int k = 0; k < 3; k++)
    {
        for (int k_ = 0; k_ < 3; k_++)
        {
            for (int l = 0; l < 3; l++)
            {
                for (int l_ = 0; l_ < 3; l_++)
                {
                    std::function<double()> zero_0;
                    if (funA.R != funB.R)
                    {

                        auto zero_0 = [&funA, &funB, k, k_, l, l_]()
                        {
                            double sigma_a = 1 / (funA.exponets[k] + funA.exponets[k_]);
                            double sigma_b = 1 / (funB.exponets[l] + funB.exponets[l_]);
                            double V_2 = 1 / (sigma_a + sigma_b);

                            double U_A = pow(M_PI * sigma_a, 3 / 2);
                            double U_B = pow(M_PI * sigma_b, 3 / 2);
                            double U = U_A * U_B;

                            double RA_RB_2 = pow(funA.R[0] + funB.R[0], 2) + pow(funA.R[1] + funB.R[1], 2) + pow(funB.R[2] + funB.R[2], 2);

                            double T = V_2 * RA_RB_2;

                            return U * pow(RA_RB_2, -1 / 2) * erf(T);
                        };
                    }
                    else
                    {

                        auto zero_0 = [&funA, &funB, k, k_, l, l_]()
                        {
                            double sigma_a = 1 / (funA.exponets[k] + funA.exponets[k_]);
                            double sigma_b = 1 / (funB.exponets[l] + funB.exponets[l_]);
                            double V_2 = 1 / (sigma_a + sigma_b);

                            double U_A = pow(M_PI * sigma_a, 3 / 2);
                            double U_B = pow(M_PI * sigma_b, 3 / 2);
                            double U = U_A * U_B;

                            return U * pow(2 * V_2, 1 / 2) * pow(2 / M_PI, 1 / 2);
                        };
                    }
                    sum += (funA.coefs[k] * funA.norms[k]) * (funA.coefs[k_] * funA.norms[k_]) * (funB.coefs[l] * funB.norms[l]) * (funB.coefs[l_] * funB.norms[l_]) * zero_0();
                }
            }
        }
    }
    return sum;
}

void CNDO2::make_gamma()
{
    vector<Function> s_basis;

    for (auto &fun : basis)
    {
        if (fun.orbital == vector<int>{0, 0, 0})
        {
            s_basis.push_back(fun);
        }
    }

    for (int i = 0; i < s_basis.size(); i++)
    {
        for (int j = 0; j < s_basis.size(); j++)
        {
            G(i, j) = gamma_AB(s_basis[i], s_basis[j]);
        }
    }
}

void CNDO2::calc_P(arma::Mat<double> &P, arma::Mat<double> &C, electrons spin)
{

    int n;
    if (spin == alpha_)
    {
        n = alpha;
    }
    else if (spin == beta_)
    {
        n = beta;
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += C(i, k) * C(j, k);
            }
            P.at(i, j) = sum;
        }
    }
}

void CNDO2::calc_f(arma::Mat<double> &f, arma::Mat<double> &P)
{

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i != j)
            {
                f.at(i, j) = 0.5 * (basis[i].B + basis[j].B) * S(i, j) - P(i, j) * gamma_AB(basis[i], basis[j]);
            }
            else
            {

                double sum = 0;
                for (auto &fun : basis)
                {
                    if (fun.R != basis[i].R)
                    {
                        sum += (P_tot(fun.atomic_num) - fun.val_elec) * gamma_AB(basis[i], fun);
                    }
                }

                f.at(i, i) = (-0.5 * (basis[i].I_A) + ((P_tot(basis[i].atomic_num) - basis[i].val_elec) - (P(i, j) - 0.5)) * gamma_AB(basis[i], basis[i])) + sum;
            }

            // should maybe write a vector to keep each atom in individually and then can expand to its basis
            //  i think i might have actually done it this way on assignment 3
            // for (int i = 0; i < num_atoms_in_total; i++)
            //     if (A != B)
            //     {
            //         f.at(i, i) += (P_BB_total - valance_atom_num_B) * gamma_AB;
            //         // here B is any but the current atom we are on
            //         // since this loop is per basis function we need to make sure somehow that this is handled appropiatly

            //         // if element number identifier, as in we can call what atom it is based on the index
            //     }
        }
    }
}

void CNDO2::calc_C(arma::Mat<double> &C, arma::Mat<double> &f)
{

    arma::Col<double> eigval;
    arma::Mat<double> eigvec;
    eig_sym(eigval, eigvec, f);

    C = eigvec;
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
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {

            S.at(i, j) = 0.0;
            Function &u = basis[i];
            Function &v = basis[j];

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
