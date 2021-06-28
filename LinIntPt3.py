import streamlit as st
import numpy as np
import pandas as pd
import sympy


from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder
st.set_page_config(layout="wide")


variable_dict = {"advanced":False, "update 11.26":False, "standard":False, "done":False}

st.title("Interior Point Algorithm for Linear Programs")
st.write("This website uses [Algorithm 11.3](https://www.wiley.com/go/veatch/convexandlinearoptimization) (Primal-dual path following) to solve a linear program in canonical form."
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
st.markdown("The standard form problem has $m$ constraints and $n′$ variables. Call the $m \\times n′$ coefficient matrix $\\bar{{A}}$ , etc.:")
st.latex(r"""\begin{aligned}
    &\text{max } \bar{c}^T\bar{x}& \\
    &\text{s.t.  } \bar{A}\bar{x} \leq b & \\
    &\bar{x} \geq 0& \end{aligned}""")
st.write("When converted to canonical form, the constraints are $\\bar{A}\\bar{x}+s=b$. Here $s$ contains $m$ slack variables and $w$ contains $n = m + n^′$ dual surplus variables. "
         "Strict feasibility for the primal requires $\\bar{x}>0, s>0$. Strict feasibility for the dual requires $w > 0$.")
st.write("Enter the problem and strictly feasible initial feasible solutions below, along with the parameters for the problem. Once you press submit, the problem is re-computed.")

variable_dict["ex 11.7"] = st.checkbox("Load Example 11.7", value=True)
if not variable_dict["ex 11.7"]:
    variable_dict['standard'] = st.checkbox("Standard form", value = False)
if not variable_dict["ex 11.7"]:
    with st.form('matrix'):
        if variable_dict["standard"]:
            st.markdown("## Input your matrix $\\bar{A}$")
            st.markdown("Write your matrix $\\bar{A}$ ($m \\times n^′$) in standard form in the top-left of this entry grid.")
        else:
            st.markdown("## Input your matrix A")
            st.write("Write your matrix A ($m \\times n$) in canonical form in the top-left of this entry grid.")
        input_dataframe = pd.DataFrame('', index=[str(i) for i in range(10)], columns=[str(i) for i in range(10)])
        grid_height = 335

        response = AgGrid(
            input_dataframe,
            height=grid_height,
            width='100%',
            editable=True,
            sortable=False,
            filter=False,
            resizable=True,
            defaultWidth=5,
            fit_columns_on_grid_load=False,
            key='input_frame')
        messy_matrix = response['data'].replace("nan", "")
        messy_matrix.replace(to_replace="", value=np.nan, inplace=True)
        messy_matrix = messy_matrix.dropna(axis=1, how='all')
        messy_matrix = messy_matrix.dropna(axis=0, how='all')
        matrix_small = np.array(messy_matrix, dtype = float)
        m_s = matrix_small.shape[0]
        n_s = matrix_small.shape[1]
        done = st.form_submit_button()
if done:
    st.write(n_s, m_s)
    st.write(matrix_small)
st.stop()
if not variable_dict["ex 11.7"] and done:
    with st.form("input"):
        st.header("Input your problem data")
        st.write("Enter vectors using spaces between entries, e.g.,\"1 4 3 2\".")
        col = st.beta_columns(2)
        col_help = 0
        #if variable_dict['standard']:
        #    help_list = [r"""$b$ is an $m$-vector""", r"""$\bar{c}$ is an $n^′$-vector""",
        #                 r"""$x$ is an $n^′$-vector""", r"""$y$ is an $m$-vector"""]
        #else:
        #    help_list = [r"""$b$ is an $m$-vector""", r"""$c$ is an $n$-vector""",
        #                 r"""$x$ is an $n$-vector""", r"""$y$ is an $m$-vector"""]
        with col[0]:
            b = np.array([float(i) for i in
                          st.text_input("Right-hand side \x1B[3m b  \x1B[0m", help = help_list[0], value="2 1").split(" ")])
        with col[1]:
            c = np.array([float(i) for i in
                          st.text_input("A c vector separated by spaces, i.e. \"1 2 0 0\"", value="1 2 0 0", help = help_list[1]).split(
                              " ")])
        st.header("Input your initial variables")
        col = st.beta_columns(2)
        with col[0]:
            x_i = [float(i) for i in
                   st.text_input("An x vector separated by spaces, i.e. \"1 .5 .5 1.5\"", value="1 .5 .5 1.5", help = help_list[2]).split(" ")]
        with col[1]:
            y_i = [float(i) for i in
                   st.text_input("A y vector separated by spaces, i.e. \"2 .5\"", value="2 .5", help = help_list[3]).split(" ")]

        st.header("Parameters")
        col = st.beta_columns(2)
        with col[0]:
            st.write(r"""$\alpha$: Step size parameter.""")
            alpha = st.number_input(r"""""", value=0.9, step=0.01, min_value=0.0, max_value=0.999,
                                        help=r"""Ensures each variable is reduced by no more than a factor of $1 - \alpha$.""")
        with col[1]:
            st.write("""$\epsilon$: Optimality tolerance.""")
            epsilon = st.number_input(r"""""", value=0.01, step=0.001, format="%f", min_value=0.00001,
                                          help=r"""Stop the algorithm once **x**$^T$**w**$< \epsilon$.""")
        with col[0]:
            st.write("""$\gamma$: Duality gap parameter.""")
            gamma = st.number_input(r"""""", value=0.25, step=0.01,
                                        help=r"""The complimentary slackness parameter $\mu$ is multiplied by $\gamma$ each iteration such that $\mu \rightarrow 0$.""")
        with col[1]:
            st.write("""$\mu$: Positive complementary slackness parameter.""")
            mu = st.number_input("", value = 5, step = 0.1)

        variable_dict["done"] = st.form_submit_button()
        #done = True
else:
    variable_dict["done"] = True
    matrix_small = np.array([[1.5, 1], [1, 1], [0, 1]])
    m_s = 3
    n_s = 2
    alpha = .9
    epsilon = .01
    gamma = .25
#if st.button("Is your matrix incorrect? Click to enter manually."):
#    matrix_input = st.text_area("Write your matrix with spaces separating the elements and a comma after each row, i.e. \"1 3 4 6, 5 3 2 1, 6 9 3 2\"", value = "1 3 4 6, 5 3 2 1, 6 9 3 2")
#    if matrix_input:
#        matrix= [i.split(" ") for i in matrix_input.split(", ")]
#        #st.write(matrix)
#        for i in range(len(matrix)):
#            for j in range(len(matrix[i])):
#               matrix[i][j] = float(matrix[i][j])
#        matrix = np.array(matrix)
#        st.write("Your manually constructed matrix is:" ,matrix)
#else:
if variable_dict["done"]:
    if not variable_dict["ex 11.7"]:
        st.header("Your data is:")
        x = np.array(x_i)
        y = np.array(y_i)
        c = np.array(c)
        matrix = np.concatenate((matrix_small, np.identity(len(y))), axis=1)
        c_1 = np.concatenate([c, np.zeros(m_s)])
        try:
            w = matrix.T.dot(y) - c
        except:
            w = matrix_small.T.dot(y) - c
        w_i = list(w)
    elif variable_dict["ex 11.7"]:
        x_i = [3, 3, 8.5, 6, 7]
        w_i = [1.0,2.0,2.0,2.0,1.0]
        b = np.array([16,12,10])
        c = np.array([4,3])
        y_i = [2.0, 2.0, 1.0]
        st.header("Example 11.7 data is:")
    st.latex("A = " + sympy.latex(sympy.Matrix(matrix_small)))
    col = st.beta_columns(5)
    col_helper1 = 0
    var = [sympy.Matrix(i) for i in [b, c, w_i, x_i, y_i]]
    names = ["b", "c", "w", "x", "y"]
    for i in range(5):
        with col[col_helper1%5]:
            st.latex(names[i] + "=" + sympy.latex(var[i]))
            col_helper1 += 1
    x = np.array(x_i)
    y = np.array(y_i)
    c = np.array(c)
    matrix = np.concatenate((matrix_small, np.identity(len(y))), axis = 1)
    c_1 = np.concatenate([c, np.zeros(m_s)])
    if not variable_dict['ex 11.7']:
        try:
            w = matrix.T.dot(y) - c
        except:
            st.write("error")
            w = matrix_small.T.dot(y) - c
        w_i = list(w)
    else:
        w = np.array(w_i)
    f= x[:n_s].dot(c[:n_s])
    variable_dict["update 11.26"] = st.checkbox("Use 11.26 to update mu?", value=False)
    if variable_dict["update 11.26"]:
        mu = gamma*np.dot(x,w)/len(x)
    #st.write("mu=",mu)
    elif variable_dict["ex 11.7"]:
        mu = 5
    else:
        mu = st.number_input("initial mu", value = 5)
iter = 0
data = []
def round_list(list, make_tuple = False):
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
            list[i] = round(list[i],4)
    return list
if variable_dict["done"]:
    if variable_dict["update 11.26"]:
        st.markdown(f"Your method of computing $\mu$ is Equation 11.26.")
    else:
        st.markdown("Your method of updating $\mu$ each iteration is $\mu^{new} = \gamma \mu$.")
    col = st.beta_columns(3)
    with col[1]:
        mu = st.number_input("initial mu (For Testing)", value = 5)
    variable_dict['advanced'] =st.checkbox("Show slacks and dual values", value = False)
    if variable_dict["advanced"]:
        #data.append(round_list([iter, tuple(x), tuple(y), tuple(w), "-", "-", "-", f, mu, x.dot(w)], make_tuple = True))
        #alist = ["k", "x", "y", "w", "dx", "dy", "dw", "f(x)", "mu", "x^Tw"]
        s = b-matrix_small.dot(x[:n_s])
        data.append(round_list([iter, mu, x.dot(w), f, x[:n_s], s, y, w[:n_s]], make_tuple=True))
        alist = ["Iteration", "mu", "Gap x^Tw", "Objective", "x", "s", "y", "w"]
    else:
        data.append(round_list([iter, mu, x.dot(w), f, x[:n_s]], make_tuple=True))
        alist = ["Iteration", "mu", "Gap x^Tw", "Objective", "x"]
    while not np.dot(x,w) < epsilon:
        diagx = np.diagflat(x)
        diagw = np.diagflat(w)
        #diagwinv = np.linalg.inv(diagw)
        diagwinv = np.array([1/i  if i != 0 else 0 for i in np.nditer(diagw)]).reshape((len(x), len(x)))
        vmu = mu*np.ones(len(x)) - diagx.dot(diagw).dot(np.ones(len(x)))
        dy = np.linalg.inv(matrix.dot(diagx).dot(diagwinv).dot(matrix.T)).dot(matrix).dot(diagwinv).dot(vmu)
        dw = matrix.T.dot(dy)
        dx = diagwinv.dot(vmu - diagx.dot(dw))
        betap = min(1, min([alpha*j for j in [-x[i]/dx[i] if dx[i] < 0 else 1000 for i in range(len(x))]]))
        betad = min(1, min([alpha*j for j in [-w[i]/dw[i] if dw[i] < 0 else 1000 for i in range(len(w))]]))
        x += betap*dx
        y += betad*dy
        w += betad*dw
        if variable_dict["update 11.26"]:
            mu = gamma*x.dot(w)/(m_s+n_s)
        else:
            mu *= gamma

        iter += 1
        f = x[:len(c)].dot(c)
        if variable_dict["advanced"]:
            #data.append(round_list([iter, x, y, w, dx, dy, dw, f, mu, x.dot(w)], make_tuple=True))
            data.append(round_list([iter, mu, x.dot(w), f, x[:n_s], s, y, w[:n_s]], make_tuple=True))
        else: #["mu", "Gap x^Tw", "Objective", "x"]
            data.append(round_list([iter, mu, x.dot(w), f, x[:n_s]], make_tuple=True))
        assert iter < 15, "Too many iterations"
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
    #st.dataframe(df)
    st.table(df)
    st.markdown("Note: The $\mu$ used in each row was used to compute that row."
                " This differs slightly from table 11.2 as here, $\mu$ is reported after it is updated.")
    col_help = 0
def latex_matrix(name, matrix_for_me, col_bool, col_use1, col_use2, col_use3, vec = True):
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
def latex_matrix(name, matrix_for_me, col_bool, col_use1, col_use2, col_use3, vec = True):
    global col_help
    latex_string = name + " = " + "\\begin{bmatrix}  "
    shape_tuple = matrix_for_me.shape
    for i in range(len(matrix_for_me)):
        if vec:
            latex_string += str  (matrix_for_me[i]) + " \\\\ "
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
        string = string + "0 &"*i + str(x[i][i]) + "  &  " + "0 & "*(x_l-i-1)
        string = string[:-3] + "\\\\ "
    string = string + "\\end{bmatrix}"
    return string
def digit_fix(subs):
    for i,j in enumerate(subs):
        if j%1 == 0:
            subs[i] = int(j)
        else:
            subs[i] = j.round(4)
            if subs[i] < 0.0001 and subs[i] > -0.0001:
                subs[i] = 0
    return(subs)
def lt(x):
    return(sympy.latex(sympy.Matrix(x)))
if st.button("Detailed output of all iterations.") and variable_dict["done"]:
    x = np.array(x_i)
    w = np.array(w_i)
    y = np.array(y_i)
    f = x[:len(c)].dot(c)
    if variable_dict["update 11.26"]:
        mu = gamma * np.dot(x, w) / len(x)
    else:
        mu = 5
    iter = 0
    while not np.dot(x, w) < epsilon:
        diagx = np.diagflat(x)
        diagw = np.diagflat(w)
        diagwinv = np.linalg.inv(diagw)
        vmu = mu * np.ones(len(x)) - diagx.dot(diagw).dot(np.ones(len(x)))
        dy = np.linalg.inv(matrix.dot(diagx).dot(diagwinv).dot(matrix.T)).dot(matrix).dot(diagwinv).dot(vmu)
        dw = matrix.T.dot(dy)
        dx = diagwinv.dot(vmu - diagx.dot(dw))
        matrix_string = ["\\mathbf{X}", "\\mathbf{W}", "\\mathbf{X}\\mathbf{W}^{-1}",
                         "\mu\\mathbf{1}", "\\mathbf{XW1}", "\\mathbf{v}(\\mu)",
                         "A", "\\mathbf{AX}\\mathbf{W}^{-1}\\mathbf{A}^T",
                         "\\mathbf{d}^x", "\\mathbf{d}^y", "\\mathbf{d}^w"]
        matrix_string = ["X", "W", "XW^{-1}",
                         "\mu1", "XW1", "v(\\mu)",
                         "A", "AXW^{-1}A^T",
                         "d^x", "d^y", "d^w"]
        complicated_eq = matrix.dot(diagx).dot(diagwinv).dot(matrix.T)
        matrix_list = round_list([np.diagflat([round(i, 4) for i in x]), np.diagflat([round(i, 4) for i in w]), diagx.dot(diagwinv).round(4),
                                  mu * np.ones(len(x)), diagx.dot(diagw).dot(np.ones(len(x))), vmu,
                                  matrix, complicated_eq, dx, dy, dw], False)

        st.write(f"Iteration {iter}")
        col = st.beta_columns(3)
        for i in range(len(matrix_string)):
            #col_help += 1

            if i in [0,1,2,6,7]:
                with col[col_help % 3]:
                    if i == 6:
                        st.latex(matrix_string[7] + "=" + sympy.latex(sympy.Matrix(complicated_eq.round(4))))
                        col_help+=2
                    elif i == 7:
                        st.latex("(" + matrix_string[7] + ")^{-1}=" + sympy.latex(sympy.Matrix(np.linalg.inv(complicated_eq).round(4))))
                        col_help+=1
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
        optionp = min([alpha * j for j in [-x[i] / dx[i] if dx[i] < 0 else 100 for i in range(len(x))]])
        optiond = min([alpha * j for j in [-w[i] / dw[i] if dw[i] < 0 else 100 for i in range(len(w))]])
        x_r = [round(i,4) for i in x]
        dx_r = [round(i,4) for i in dx]
        dw_r = [round(i,4) for i in dw]
        w_r = [round(i,4) for i in w]
        betap = min(1, optionp)
        betad = min(1, optiond)
        l_string = f"\\beta_P = \\text{{min}}(1, 0.9*\\text{{min}}("
        for i in range(len(x)):
            if dx_r[i]<0:
                l_string+= "\\frac{" + str(x_r[i]) +"}{"+ str(-dx_r[i]) + "},"
        l_string = l_string[:-1] +  f") = \\text{{min}}(1, {round(optionp,4)}) = {round(betap,4)}"
        st.latex(l_string)
        l_string = f"\\beta_D = \\text{{min}}(1, 0.9*\\text{{min}}("
        for i in range(len(x)):
            if dw_r[i]<0:
                l_string+= "\\frac{" + str(w_r[i]) +"}{"+ str(-dw_r[i]) + "},"
        l_string = l_string[:-1] +  f") = \\text{{min}}(1, {round(optiond,4)}) = {round(betad,4)}"
        st.latex(l_string)
        col = st.beta_columns(3)
        with col[0]:
            st.latex("x^{new} =" + lt(digit_fix(x)) + "+" + str(round(betap,4)) + lt(digit_fix(dx)) + " = " + lt(digit_fix(x + betap * dx)))
        with col[1]:
            st.latex("y^{new} =" + lt(digit_fix(y))+ "+" + str(round(betad,4)) + lt(digit_fix(dy)) + " = " + lt(digit_fix(y + betad * dy)))
        with col[2]:
            st.latex("w^{new} ="+ lt(digit_fix(w)) + "+" + str(round(betad,4))+ lt(digit_fix(dw)) + " = " + lt(digit_fix(w + betad * dw)))
        x += betap * dx
        y += betad * dy
        w += betad * dw
        mu *= gamma
        iter += 1
        st.write("""---""")
        assert iter <= len(df), "Too many iterations"
st.markdown('''
#### Coded by [Abraham Holleran](https://github.com/Stonepaw90) :sunglasses:
''')
