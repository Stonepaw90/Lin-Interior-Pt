import streamlit as st
import numpy as np
import pandas as pd


from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder
st.set_page_config(layout="wide")


variable_dict = {"ex 11.7":False, "advanced":False, "update 11.26":False}

st.sidebar.header("Parameters")
st.sidebar.write(r"""$\alpha$: Step size parameter.""")
alpha = st.sidebar.number_input(r"""""", value = 0.9, step=0.01,min_value = 0.0, max_value = 0.999, help = r"""Ensures each variable is reduced by no more than a factor of $1 - \alpha$.""")
st.sidebar.markdown("""---""")
st.sidebar.write("""$\epsilon$: Optimality tolerance.""")
epsilon = st.sidebar.number_input(r"""""", value = 0.01, step=0.001, format="%f", min_value = 0.00001, help = r"""Stop the algorithm once **x**$^T$**w**$< \epsilon$.""")
st.sidebar.markdown("""---""")
st.sidebar.write("""$\gamma$: Duality gap parameter.""")
gamma = st.sidebar.number_input(r"""""", value = 0.25, step=0.01, help = r"""The complimentary slackness parameter $\mu$ is multiplied by $\gamma$ each iteration such that $\mu \rightarrow 0$.""")
#st.sidebar.markdown("""---""")
st.title("Primal-dual Interior Point Algorithm for Linear Programs")
st.write("Use this website to solve problems with the Primal-dual Interior Point Algorithm for Linear Programs from [Section 11.3](https://www.wiley.com/go/veatch/convexandlinearoptimization).")
#variable_dict["advanced"] = st.sidebar.checkbox("Advanced output", value = False)
st.sidebar.markdown('''
#### Coded by [Abraham Holleran](https://github.com/Stonepaw90) :sunglasses:
''')
variable_dict["ex 11.7"] = st.checkbox("Follow along with Example. 11.7", value=True)
#c1, c2 = st.beta_columns(2)
#with c2:
#    n_s = st.number_input("How many rows in your standard form A matrix?", value = 3)
#with c1:
#    m_s = st.number_input("How many columns in your standard form A matrix?", value = 4)
    #st.sidebar.number_input("Grid height", min_value=200, max_value=800, value=200)


if not variable_dict["ex 11.7"]:
    input_dataframe = pd.DataFrame('', index=range(10), columns=["C" + str(i) for i in range(10)])
    grid_height = 335
    st.write("Write your matrix A in standard form in the top-left of this entry grid.")
    c1, c2 = st.beta_columns(2)
    with c1:
        response = AgGrid(
            input_dataframe,
            height=grid_height,
            width='100%',
            editable=True,
            sortable=False,
            filter=False,
            resizable=True,
            defaultWidth=10,
            fit_columns_on_grid_load=False,
            key='input_frame')
    response['data'] = response['data'].replace("nan", "")
    n_s = len([i for i in response['data'].loc[1] if i])
    m_s = len([i for i in response['data']["C0"] if i])
    coln = ["C" + str(i) for i in range(m_s)]
    rown = [str(i) for i in range(n_s)]
    new_df = response['data'].loc[0:(m_s-1)].iloc[:,0:(n_s)].dropna()

if variable_dict["ex 11.7"]:
    matrix_small = np.array([[1.5, 1], [1, 1], [0, 1]])
    m_s = 3
    n_s = 2
else:
    matrix_small = np.array(new_df)
st.write("Your matrix is",matrix_small)
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
st.header("Input your problem data")
col = st.beta_columns(2)
col_help = 0
with col[0]:
    b = np.array([float(i) for i in st.text_input("A b vector separated by spaces, i.e. \"2 1\"", value = "2 1").split(" ")])
with col[1]:
    c = np.array([float(i) for i in st.text_input("A c vector separated by spaces, i.e. \"1 2 0 0\"", value = "1 2 0 0").split(" ")])
st.header("Input your initial variables")
col = st.beta_columns(2)
with col[0]:
    x_i = [float(i) for i in st.text_input("An x vector separated by spaces, i.e. \"4 3 1 9\"", value = "4 3 1 9").split(" ")]
with col[1]:
    y_i = [float(i) for i in st.text_input("A y vector separated by spaces, i.e. \"2 .5\"", value = "2 .5").split(" ")]



if variable_dict["ex 11.7"]:
    x_i = [3, 3, 8.5, 6, 7]
    w_i = [1.0,2.0,2.0,2.0,1.0]
    b = np.array([16,12,10])
    c = np.array([4,3])
    y_i = [2.0, 2.0, 1.0]
#s = b - matrix.dot(x_i[:len(c)])
x = np.array(x_i)
y = np.array(y_i)
c = np.array(c)
matrix = np.concatenate((matrix_small, np.identity(len(y))), axis = 1)
c_1 = np.concatenate([c, np.zeros(m_s)])
w_i = list(matrix.T.dot(y)-c_1)
w = np.array(w_i)
f= x[:n_s].dot(c[:n_s])
#y = np.array([2,.5])
if variable_dict["update 11.26"]:
    mu = gamma*np.dot(x,w)/len(x)
#st.write("mu=",mu)
elif variable_dict["ex 11.7"]:
    mu = 5
else:
    mu = st.number_input("mu", value = 5)
iter = 0
data = []
def round_list(list, make_tuple = False):
    for i in range(len(list)):
        if type(list[i]) is str or type(list[i]) is tuple:
            pass
        elif type(list[i]) is list or type(list[i]) is np.ndarray:
            try:
                for j in range(len(list[i])):
                    list[i][j] = round(list[i][j], 3)
                if make_tuple:
                    list[i] = tuple(list[i])
            except:
                pass
        else:
            list[i] = round(list[i],3)
    return list
if not variable_dict["ex 11.7"]:
    variable_dict["update 11.26"] = st.checkbox("Use 11.26 to update mu?", value = False)
else:
    variable_dict["update 11.26"] = True
if variable_dict["update 11.26"]:
    st.markdown(f"Your method of computing $\mu$ is Equation 11.26.")
else:
    st.markdown("Your method of computing $\mu$ is $\mu^{new} = \gamma \mu$.")

variable_dict['advanced'] =st.checkbox("Show Expanded Output", value = False)

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
    diagwinv = np.linalg.inv(diagw)
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
st.markdown("Note: The $\mu$ used in each row was used to compute that row. This differs slightly from table 11.2.")
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
def diagonal_matrix(x):
    string = f"\\begin{{bmatrix}}"
    x_l = len(x)
    for i in range(x_l):
        string = string + "0 &"*i + str(x[i][i]) + "  &  " + "0 & "*(x_l-i-1)
        string = string[:-3] + "\\\\ "
    string = string + "\\end{bmatrix}"
    return string
if st.button("Show all the matrixes please.") or True:
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
        matrix_string = ["\\mathbf{v}(\\mu)", "\\mathbf{X}", "\\mathbf{W}",
                         "\\mathbf{d}^x", "\\mathbf{d}^y", "\\mathbf{d}^w"]
        matrix_list = round_list([vmu, np.diagflat([round(i, 4) for i in x]), np.diagflat([round(i, 4) for i in w]), dx, dy, dw], False)
        col = st.beta_columns(3)
        for i in range(len(matrix_string)):
            #col_help += 1
            if i == 1 or i == 2:
                with col[col_help % 3]:
                    st.latex(matrix_string[i] + "=" + diagonal_matrix(matrix_list[i]))
                    col_help += 1
                #if i == 2:
                    #st.write("Solving (11.22),")
            else:
                latex_matrix(matrix_string[i], matrix_list[i], True, col[0], col[1], col[2])
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
        x += betap * dx
        y += betad * dy
        w += betad * dw
        mu *= gamma
        iter += 1
        st.write("""---""")
        assert iter <= len(df), "Too many iterations"
