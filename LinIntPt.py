import streamlit as st
import numpy as np
import pandas as pd
import sympy

from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder

st.set_page_config(layout="wide")

variable_dict = {"advanced": False, "update 11.26": False, "standard": False, "done": False}

st.title("Interior Point Algorithm for Linear Programs")
st.write(
    "This website uses [Algorithm 11.3](https://www.wiley.com/go/veatch/convexandlinearoptimization) (Primal-dual path following) to solve a linear program in canonical form."
    " If the problem is entered in standard form, it is converted to canonical form.")
st.header("Standard and canonical form notation")
st.markdown("The canonical form problem has $m$ constraints and $n$ variables:")
col = st.beta_columns(2)
with col[0]:
    st.write("Primal")
    st.latex(r"""\begin{aligned}
    &\text{max } c^Tx& \\
    &\text{s.t.  } Ax = b & \\
    &x \geq 0& \end{aligned}""")
with col[1]:
    st.write("Dual")
    st.latex(r"""\begin{aligned}
    &\text{min } b^Ty& \\
    &\text{s.t.  } A^Ty -w = c & \\
    &w \geq 0& \end{aligned}""")
st.markdown(
    "The standard form problem has $m$ constraints and $n′$ variables. Call the $m \\times n′$ coefficient matrix $\\bar{{A}}$ , etc.:")
st.latex(r"""\begin{aligned}
    &\text{max } \bar{c}^T\bar{x}& \\
    &\text{s.t.  } \bar{A}\bar{x} \leq b & \\
    &\bar{x} \geq 0& \end{aligned}""")
st.write(
    "When converted to canonical form, the constraints are $\\bar{A}\\bar{x}+s=b$. Here $s$ contains $m$ slack variables and $w$ contains $n = m + n^′$ dual surplus variables. "
    "Strict feasibility for the primal requires $\\bar{x}>0, s>0$. Strict feasibility for the dual requires $w > 0$.")
st.write(
    "Enter the problem and strictly feasible initial feasible solutions below, along with the parameters for the problem. The problem is re-solved after any changes.")

variable_dict["ex 11.7"] = st.checkbox("Load Example 11.7", value=True)
input_dataframe = pd.DataFrame('', index=[str(i + 1) for i in range(10)], columns=[str(i + 1) for i in range(10)])
error = False
# If we're writing our own data:
if not variable_dict["ex 11.7"]:
    # You can choose if standard or not standard.
    variable_dict['standard'] = st.checkbox("Standard form", value=False)
    st.markdown("## Coefficient matrix A")
    # Standard
    if variable_dict["standard"]:
        st.markdown("Write your matrix A ($m \\times n^′$) in standard form in the top-left of this entry grid."
                    " The program will append the identity matrix, converting everything to canonical form.")
    # Canonical
    else:
        st.write("Write your matrix A ($m \\times n$) in canonical form in the top-left of this entry grid.")
    st.write(
        "Warning: Submitting while a box is being edited will return an error. Before submitting, press enter or tab to confirm your edits.")
    grid_height = 335
    response = AgGrid(
        input_dataframe,
        height=grid_height,
        width='100%',
        editable=True,
        sortable=False,
        filter=False,
        resizable=True,
        defaultWidth=15,
        fit_columns_on_grid_load=False,
        key='input_frame')
    # Convert Matrix, catching errors. Errors lead to a stop that prints out the matrix and your matrix shape (m_s, n_s).
    try:
        messy_matrix = response['data'].replace("nan", "")
        messy_matrix.replace(to_replace="", value=np.nan, inplace=True)
        messy_matrix = messy_matrix.dropna(axis=1, how='all')
        messy_matrix = messy_matrix.dropna(axis=0, how='all')
        matrix_small = np.array(messy_matrix, dtype=float)
        m_s = matrix_small.shape[0]
        n_s = matrix_small.shape[1]
        # If the matrix contains any NaNs, move to the except clause.
        assert not np.isnan(matrix_small).any()
    except:  # If any errors
        st.write("Something is wrong with your matrix. ")
        try:  # Try to give diagnostics
            st.latex("A = " + sympy.latex(sympy.Matrix(matrix_small)))
            st.write("It has shape (", m_s, "*", n_s, ")")
        except:
            pass
        st.write(" Please ensure dimensions and entries are correct and submit again.")
        st.stop()  # Nice exit of program, with no giant red errors.
    if m_s > 0 and n_s > 0:
        # Only get here if no errors. #Non-zero matrix, with no errors! Yay!
        st.write("This is your matrix. It has shape (", m_s, "*", n_s, ") If this is incorrect, submit again.")
        st.latex("A = " + sympy.latex(sympy.Matrix(matrix_small)))
        st.header("Input your problem data")
        st.write("Enter vectors using spaces between entries, e.g.,\"1 4 3 2\".")
        col = st.beta_columns(2)
        # col_help = 0
        with col[0]:
            b = np.array([float(i) for i in
                          st.text_input(f"Right-hand side b (a {m_s}-vector)", value="2 1").split(" ")])
        if variable_dict["standard"]:
            n_full = n_s + m_s
        else:
            n_full = n_s
        with col[1]:
            c = np.array([float(i) for i in
                          st.text_input(f"Objective function coefficients c (a {n_full}-vector)", value="1 2").split(
                              " ")])
        st.header("Initial solution")
        col = st.beta_columns(2)
        with col[0]:
            x = np.array([float(i) for i in
                          st.text_input(f"x (a {n_full}-vector)", value="1 .5").split(" ")])
        with col[1]:
            y = np.array([float(i) for i in
                          st.text_input(f"y (a {m_s}-vector)", value="2 .5").split(" ")])
        st.header("Parameters")
        col = st.beta_columns(2)
        with col[0]:
            st.write(r"""$\alpha$: Step size parameter.""")
            alpha = st.number_input(r"""""", value=0.9, step=0.01, min_value=0.0, max_value=0.999,
                                    help=r"""Ensures each variable is reduced by no more than a factor of $1 - \alpha$.""")
            st.write("""$\gamma$: Duality gap parameter.""")
            gamma = st.number_input(r"""""", value=0.25, step=0.01,
                                    help=r"""The complimentary slackness parameter $\mu$ is multiplied by $\gamma$ each iteration such that $\mu \rightarrow 0$.""")
        with col[1]:
            st.write("""$\epsilon$: Optimality tolerance.""")
            epsilon = st.number_input(r"""""", value=0.01, step=0.001, format="%f", min_value=0.00001,
                                      help=r"""Stop the algorithm once **x**$^T$**w**$< \epsilon$.""")
            st.write("""$\mu$: Initial complementary slackness parameter.""")
            mu = st.number_input("", value=5.0, step=0.1)
        variable_dict["done"] = st.checkbox("Are all your variables correct?")
        # In this case, it's not var["ex 11.7"] so I could put "your data is:" but IDK about checkbox
        ###TODO
else:  # In this case, it's 11.7 data that we're loading.
    variable_dict["done"] = True
    variable_dict["standard"] = True
    matrix_small = np.array([[1.5, 1], [1, 1], [0, 1]])
    m_s = 3
    n_s = 2
    n_full = n_s + m_s
    alpha = .9
    epsilon = .01
    gamma = .25
    mu = 5.0
    x = np.array([3, 3])  # , 8.5, 6, 7]
    # w_i = [1.0, 2.0, 2.0, 2.0, 1.0]
    b = np.array([16, 12, 10])
    c = np.array([4, 3])
    y = np.array([2.0, 2.0, 1.0])
    st.header("Example 11.7 data is:")


def is_neg(x):
    return any([i <= 0 for i in x])


if variable_dict["done"]:  # Or if ex 11.7
    # Always run! Ex 11.7, standard, canonical, this is always run.
    # By this point, crucially, are data has been marked as correct. We should still check this.
    if not variable_dict["ex 11.7"]:
        st.header("Your data is:")

    if variable_dict["standard"]:
        s = b - matrix_small.dot(x)
        matrix_full = np.concatenate((matrix_small, np.identity(m_s)), axis=1)
        x_full = np.concatenate((x, s))
        c_full = np.concatenate((c, np.zeros(m_s)))
    else:
        matrix_full = matrix_small
        x_full = x
        c_full = c
            # matrix_full = np.concatenate((matrix_small, np.identity(m_s)), axis=1)
            # x_full = np.concatenate((x,s))
            # c_full = np.concatenate((c, np.zeros(m_s)))
    w = matrix_full.T.dot(y) - c_full
    #I don't know why this was so difficult! I'm saving these initial values for later.
    w_initial = list(w)
    x_initial = list(x_full)
    y_initial = list(y)
    mu_initial = mu/2
    st.latex("A = " + sympy.latex(sympy.Matrix(matrix_small)))
    col = st.beta_columns(5)
    col_helper1 = 0
    var = [sympy.Matrix(i) for i in [b, c_full, w, x_full, y]]
    names = ["b", "c", "w", "x", "y"]
    for i in range(5):
        with col[col_helper1 % 5]:
            st.latex(names[i] + "=" + sympy.latex(var[i]))
            col_helper1 += 1
    if any([is_neg(x_full), is_neg(w)]):
        st.write(f"One of your vectors is negative. Your vectors are x = {x_full}, w = {w}")
        st.stop()
    f = x_full.dot(c_full)
    variable_dict["update 11.26"] = st.checkbox("Use 11.26 to update mu?", value=False)
    if variable_dict["update 11.26"]:
        st.markdown(f"Your method of computing $\mu$ is Equation 11.26.")
    else:
        st.markdown("Your method of updating $\mu$ each iteration is $\mu^{new} = \gamma \mu$.")
    # if variable_dict["update 11.26"]:
    #    mu = gamma*np.dot(x,w)/len(x)
    # st.write("mu=",mu)
    # elif variable_dict["ex 11.7"]:
    #    mu = 5
iter = 0
data = []


def round_list(list, make_tuple=False):
    for i in range(len(list)):
        if type(list[i]) is str or type(list[i]) is tuple:
            pass
        elif type(list[i]) is list or type(list[i]) is np.ndarray:
            try:
                for j in range(len(list[i])):
                    list[i][j] = round(list[i][j], 4)
                if make_tuple:
                    list[i] = tuple(list[i])
            except:
                pass
        else:
            list[i] = round(list[i], 4)
    return list


if variable_dict["done"]:  # All branches get here, once data has been verified.
    variable_dict['advanced'] = st.checkbox("Show slacks and dual values", value=False)

    ###ITERATION 0 ROW
    if variable_dict["advanced"]:
        # IN CANONICAL FORM THERE IS NO S TO PRINT! We're already printing x_full.
        if variable_dict["standard"]:
            data.append(round_list([iter, mu, x_full.dot(w), f, x, s, y, w], make_tuple=True))
            alist = ["Iteration", "mu", "Gap x^Tw", "Objective", "x", "s", "y", "w"]
        else:
            data.append(round_list([iter, mu, x_full.dot(w), f, x_full, y, w], make_tuple=True))
            alist = ["Iteration", "mu", "Gap x^Tw", "Objective", "x", "y", "w"]
    else:
        if variable_dict["standard"]:  # Not Advanced, and Standard
            data.append(round_list([iter, mu, x_full.dot(w), f, x], make_tuple=True))
            alist = ["Iteration", "mu", "Gap x^Tw", "Objective", "x"]
        else:  # Not advanced, canonical
            data.append(round_list([iter, mu, x_full.dot(w), f, x_full], make_tuple=True))
            alist = ["Iteration", "mu", "Gap x^Tw", "Objective", "x"]

    while not np.dot(x_full, w) < epsilon:
        diagx = np.diagflat(x_full)
        diagw = np.diagflat(w)
        # diagwinv = np.linalg.inv(diagw)
        diagwinv = np.array([1 / i if i != 0 else 0 for i in np.nditer(diagw)]).reshape((n_full, n_full))
        vmu = mu * np.ones(n_full) - diagx.dot(diagw).dot(np.ones(n_full))
        dy = np.linalg.inv(matrix_full.dot(diagx).dot(diagwinv).dot(matrix_full.T)).dot(matrix_full).dot(diagwinv).dot(vmu)
        dw = matrix_full.T.dot(dy)
        dx = diagwinv.dot(vmu - diagx.dot(dw))
        betap = min(1, min([alpha * j for j in [-x_full[i] / dx[i] if dx[i] < 0 else 1000 for i in range(n_full)]]))
        betad = min(1, min([alpha * j for j in [-w[i] / dw[i] if dw[i] < 0 else 1000 for i in range(n_full)]]))
        x_full += betap * dx
        y += betad * dy
        w += betad * dw
        if variable_dict["update 11.26"]:
            mu = gamma * x.dot(w) / (m_s + n_s)
        else:
            mu *= gamma

        iter += 1
        f = x_full.dot(c_full)
        if variable_dict["advanced"]:
            if variable_dict["standard"]:
                data.append(round_list([iter, mu, x_full.dot(w), f, x_full[:n_s], s, y, w], make_tuple=True))
            else:
                data.append(round_list([iter, mu, x_full.dot(w), f, x_full, y, w], make_tuple=True))
        else:
            if variable_dict["standard"]:  # Not Advanced, and Standard
                data.append(round_list([iter, mu, x_full.dot(w), f, x_full[:n_s]], make_tuple=True))
            else:  # Not advanced, canonical
                data.append(round_list([iter, mu, x_full.dot(w), f, x_full], make_tuple=True))

        assert iter < 15, "The program terminated, as after 15 iterations, the duality gap was still more than epsilon."
    df = pd.DataFrame(data, columns=alist)
    st.markdown("""
    <style>
    table td:nth-child(1) {
        display: none
    }
    table th:nth-child(1) {
        display: none
    }
    </style>
    """, unsafe_allow_html=True)
    # st.dataframe(df)
    st.table(df)
    st.markdown("Note: The $\mu$ used in each row was used to compute that row."
                " This differs slightly from table 11.2 as here, $\mu$ is reported after it is updated.")
    col_help = 0


def latex_matrix(name, matrix_for_me, col_bool, col_use1, col_use2, col_use3, vec=True):
    global col_help
    latex_string = name + " = " + "\\begin{bmatrix}  "
    shape_tuple = matrix_for_me.shape
    for i in range(len(matrix_for_me)):
        if vec:
            latex_string += str(matrix_for_me[i]) + " \\\\ "
        else:
            latex_string += str(matrix_for_me[i]) + " & "
        if len(shape_tuple) > 1:
            if ((i + 1) % shape_tuple[1] == 0):
                latex_string = latex_string[:-3] + " \\\\ "
    latex_string = latex_string[:-3] + "  \\end{bmatrix}"
    try:
        if col_bool:
            if col_help % 3 == 0:
                with col_use1:
                    st.latex(latex_string)
            elif col_help % 3 == 1:
                with col_use2:
                    st.latex(latex_string)
            else:
                with col_use3:
                    st.latex(latex_string)
        else:
            st.latex(latex_string)
    except:
        st.write("Something broke while trying to print a matrix.")
    col_help += 1


def latex_matrix(name, matrix_for_me, col_bool, col_use1, col_use2, col_use3, vec=True):
    global col_help
    latex_string = name + " = " + "\\begin{bmatrix}  "
    shape_tuple = matrix_for_me.shape
    for i in range(len(matrix_for_me)):
        if vec:
            latex_string += str(matrix_for_me[i]) + " \\\\ "
        else:
            latex_string += str(matrix_for_me[i]) + " & "
        if len(shape_tuple) > 1:
            if ((i + 1) % shape_tuple[1] == 0):
                latex_string = latex_string[:-3] + " \\\\ "
    latex_string = latex_string[:-3] + "  \\end{bmatrix}"
    try:
        if col_bool:
            if col_help % 3 == 0:
                with col_use1:
                    st.latex(latex_string)
            elif col_help % 3 == 1:
                with col_use2:
                    st.latex(latex_string)
            else:
                with col_use3:
                    st.latex(latex_string)
        else:
            st.latex(latex_string)
    except:
        st.write("Something broke while trying to print a matrix.")
    col_help += 1


def diagonal_matrix(x):
    string = f"\\begin{{bmatrix}}"
    x_l = len(x)
    for i in range(x_l):
        string = string + "0 &" * i + str(x[i][i]) + "  &  " + "0 & " * (x_l - i - 1)
        string = string[:-3] + "\\\\ "
    string = string + "\\end{bmatrix}"
    return string


def digit_fix(subs):
    for i, j in enumerate(subs):
        if j % 1 == 0:
            subs[i] = int(j)
        else:
            subs[i] = j.round(4)
            if subs[i] < 0.0001 and subs[i] > -0.0001:
                subs[i] = 0
    return (subs)


def lt(x):
    return (sympy.latex(sympy.Matrix(x)))


#if st.button("Detailed output of all iterations.") and variable_dict["done"]:
if variable_dict["done"]:
    w = np.array(w_initial)
    x_full = np.array(x_initial)
    y = np.array(y_initial)
    mu = mu_initial*2


    f = x.dot(c)
    # if variable_dict["update 11.26"]:
    #    mu = gamma * np.dot(x, w) / len(x)
    # else:
    #    mu = 5
    iter = 0
    st.write("Detailed output of all iterations below.")
    st.write("# ")
    st.write("""---""")
    st.write("# ")
    while not np.dot(x_full, w) < epsilon:
        diagx = np.diagflat(x_full)
        diagw = np.diagflat(w)
        diagwinv = np.array([1 / i if i != 0 else 0 for i in np.nditer(diagw)]).reshape((n_full, n_full))
        vmu = mu * np.ones(n_full) - diagx.dot(diagw).dot(np.ones(n_full))
        dy = np.linalg.inv(matrix_full.dot(diagx).dot(diagwinv).dot(matrix_full.T)).dot(matrix_full).dot(diagwinv).dot(vmu)
        dw = matrix_full.T.dot(dy)
        dx = diagwinv.dot(vmu - diagx.dot(dw))
        matrix_string = ["\\mathbf{X}", "\\mathbf{W}", "\\mathbf{X}\\mathbf{W}^{-1}",
                         "\mu\\mathbf{1}", "\\mathbf{XW1}", "\\mathbf{v}(\\mu)",
                         "A", "\\mathbf{AX}\\mathbf{W}^{-1}\\mathbf{A}^T",
                         "\\mathbf{d}^x", "\\mathbf{d}^y", "\\mathbf{d}^w"]
        matrix_string = ["X", "W", "XW^{-1}",
                         "\mu1", "XW1", "v(\\mu)",
                         "A", "AXW^{-1}A^T",
                         "d^x", "d^y", "d^w"]
        complicated_eq = matrix_full.dot(diagx).dot(diagwinv).dot(matrix_full.T)
        matrix_list = round_list([np.diagflat([round(i, 4) for i in x_full]), np.diagflat([round(i, 4) for i in w]),
                                  diagx.dot(diagwinv).round(4),
                                  mu * np.ones(n_full), diagx.dot(diagw).dot(np.ones(n_full)), vmu,
                                  matrix_full, complicated_eq, dx, dy, dw], False)

        st.write(f"Iteration {iter}")
        col = st.beta_columns(3)
        for i in range(len(matrix_string)):
            # col_help += 1

            if i in [0, 1, 2, 6, 7]:
                with col[col_help % 3]:
                    if i == 6:
                        st.latex(matrix_string[7] + "=" + sympy.latex(sympy.Matrix(complicated_eq.round(4))))
                        col_help += 2
                    elif i == 7:
                        st.latex("(" + matrix_string[7] + ")^{-1}=" + sympy.latex(
                            sympy.Matrix(np.linalg.inv(complicated_eq).round(4))))
                        col_help += 1
                    else:
                        st.latex(matrix_string[i] + "=" + diagonal_matrix(matrix_list[i]))
                        col_help += 1
            else:
                latex_matrix(matrix_string[i], matrix_list[i], True, col[0], col[1], col[2])
            if i == 2:
                st.write("Details of (11.22):")
                col = st.beta_columns(3)
                col_help = 2
            if i == 5:
                st.write("Details of (11.23):")
                col = st.beta_columns(3)
                col_help = 0
            if i == 7:
                st.markdown("Solving for **d**:")
                col = st.beta_columns(3)
                col_help = 0

        st.write("The step sizes are")
        optionp = min([alpha * j for j in [-x_full[i] / dx[i] if dx[i] < 0 else 100 for i in range(n_full)]])
        optiond = min([alpha * j for j in [-w[i] / dw[i] if dw[i] < 0 else 100 for i in range(n_full)]])
        x_r = [round(i, 4) for i in x_full]
        dx_r = [round(i, 4) for i in dx]
        dw_r = [round(i, 4) for i in dw]
        w_r = [round(i, 4) for i in w]
        betap = min(1, optionp)
        betad = min(1, optiond)
        l_string = f"\\beta_P = \\text{{min}}(1, 0.9*\\text{{min}}("
        for i in range(n_full):
            if dx_r[i] < 0:
                l_string += "\\frac{" + str(x_r[i]) + "}{" + str(-dx_r[i]) + "},"
        l_string = l_string[:-1] + f") = \\text{{min}}(1, {round(optionp, 4)}) = {round(betap, 4)}"
        st.latex(l_string)
        l_string = f"\\beta_D = \\text{{min}}(1, 0.9*\\text{{min}}("
        for i in range(n_full):
            if dw_r[i] < 0:
                l_string += "\\frac{" + str(w_r[i]) + "}{" + str(-dw_r[i]) + "},"
        l_string = l_string[:-1] + f") = \\text{{min}}(1, {round(optiond, 4)}) = {round(betad, 4)}"
        st.latex(l_string)
        col = st.beta_columns(3)
        with col[0]:
            st.latex("x^{new} =" + lt(digit_fix(x_full)) + "+" + str(round(betap, 4)) + lt(digit_fix(dx)) + " = " + lt(
                digit_fix(x_full + betap * dx)))
        with col[1]:
            st.latex("y^{new} =" + lt(digit_fix(y)) + "+" + str(round(betad, 4)) + lt(digit_fix(dy)) + " = " + lt(
                digit_fix(y + betad * dy)))
        with col[2]:
            st.latex("w^{new} =" + lt(digit_fix(w)) + "+" + str(round(betad, 4)) + lt(digit_fix(dw)) + " = " + lt(
                digit_fix(w + betad * dw)))
        x_full += betap * dx
        y += betad * dy
        w += betad * dw
        mu *= gamma
        iter += 1
        st.write("""---""")
        assert iter <= len(df), "Too many iterations"
st.markdown('''
#### Coded by [Abraham Holleran](https://github.com/Stonepaw90) :sunglasses:
''')
