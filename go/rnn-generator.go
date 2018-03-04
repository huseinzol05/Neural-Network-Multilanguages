package main

import ("encoding/csv"
        "fmt"
        "os"
        "gonum.org/v1/gonum/mat"
        "strconv"
        "math"
        "math/rand")

func removeDuplicates(elements []string) []string {
    encountered := map[string]bool{}
    result := []string{}
    for v := range elements {
        if encountered[elements[v]] == true {
        } else {
            encountered[elements[v]] = true
            result = append(result, elements[v])
        }
    }
    return result
}

func printMatrix(X *mat.Dense){
    fm := mat.Formatted(X, mat.Prefix("    "), mat.Squeeze())
    fmt.Printf("m = %4.2f", fm)
}

func indexOf(element string, elements []string) int{
    for i := 0; i < len(elements); i++ {
        if elements[i] == element {
            return i
        }
    }
    return -1
}

func generateConstant(x int, y int, constant float64) *mat.Dense{
    size := x * y
    matrix := make([]float64, size)
    for i := 0; i < size; i++ {
        matrix[i] = constant
    }
    return mat.NewDense(x, y, matrix)
}

func sigmoid(x float64, grad bool) float64{
    if grad{
        return sigmoid(x,false) * (1.0 - sigmoid(x,false))
    } else {
        return 1.0 / (1.0 + math.Exp(-x))
    }
}

func tanh(x float64, grad bool) float64{
    if grad{
        output := tanh(x,false)
        return (1.0 - output * output)
    } else{
        return math.Tanh(x)
    } 
}

func cross_entropy(x float64, y float64, grad bool) float64{
    if grad{
        return -(y / x) + (1 - y) / (1 - x)
    } else{
        return -y * math.Log(x) - (1 - y) * math.Log(1 - x)
    }
}

func calculateCrossEntropy(X *mat.Dense, Y *mat.Dense, grad bool) *mat.Dense{
    rows, columns := X.Dims()
    data := make([]float64, rows * columns)
    for i := 0; i < rows; i++ {
        for k := 0; k < columns; k++ {
            data[(i*columns)+k] = cross_entropy(X.At(i, k), Y.At(i, k), grad)
        }
    }
    return mat.NewDense(rows, columns, data)
}

func calculateTanh(X *mat.Dense, grad bool) *mat.Dense{
    rows, columns := X.Dims()
    data := make([]float64, rows * columns)
    for i := 0; i < rows; i++ {
        for k := 0; k < columns; k++ {
            data[(i*columns)+k] = tanh(X.At(i, k), grad)
        }
    }
    return mat.NewDense(rows, columns, data)
}

func getMaxColumn(X *mat.Dense) *mat.Dense{
    rows, columns := X.Dims()
    data := make([]float64, rows)
    for i := 0; i < rows; i++ {
        data[i] = mat.Max(X.RowView(i))
    }
    m := generateConstant(rows, columns, 0)
    for i := 0; i < columns; i++ {
        m.SetCol(i, data)
    }
    return m
}

func getSumColumn(X *mat.Dense) *mat.Dense{
    rows, columns := X.Dims()
    data := make([]float64, rows)
    for i := 0; i < rows; i++ {
        data[i] = mat.Sum(X.RowView(i))
    }
    m := generateConstant(rows, columns, 0)
    for i := 0; i < columns; i++ {
        m.SetCol(i, data)
    }
    return m
}

func softmax(X *mat.Dense, grad bool) *mat.Dense{
    if grad{
        p := softmax(X, false)
        rows, columns := p.Dims()
        ones := generateConstant(rows, columns, 1)
        ones.Sub(ones, p)
        p.MulElem(p, ones)
        return p
    } else{
        max_columns := getMaxColumn(X)
        X.Sub(X, max_columns)
        applyExp := func(_, _ int, v float64) float64 { return math.Exp(v) }
        X.Apply(applyExp, X)
        sum_columns := getSumColumn(X)
        X.DivElem(X, sum_columns)
        return X
    }
}

func derivative_cross_entropy_softmax(X *mat.Dense, Y []int) *mat.Dense{
    rows, columns := X.Dims()
    for i := 0; i < rows; i++ {
        X.Set(i, Y[i], X.At(i, Y[i]) - 1)
    }
    return X
}

func indexOfMax(arr []float64) int{
    max := arr[0];
    maxIndex := 0;
    for i := 1; i < len(arr); i++ {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    return maxIndex;
}

func getAccuracy(X *mat.Dense, Y *mat.Dense) float64{
    correct := 0
    rows, _ := X.Dims()
    for i := 0; i < rows; i++ {
        x := indexOfMax(X.RawRowView(i))
        y := indexOfMax(Y.RawRowView(i))
        if x == y{
            correct++
        }
    }
    return float64(correct) / float64(rows)
}

func randomMatrix(x int, y int) *mat.Dense{
    data := make([]float64, x * y)
    for i := range data {
        data[i] = rand.NormFloat64()
    }
    return mat.NewDense(x, y, data)
}

func forward_multiply_gate(w *mat.Dense, x *mat.Dense) *mat.Dense{
    var a mat.Dense
    a.Mul(w, x)
    return &a
}

func backward_multiply_gate(w *mat.Dense, x *mat.Dense, dz *mat.Dense) (*mat.Dense, *mat.Dense){
    var dw mat.Dense
    var dx mat.Dense
    dw.Mul(dz.T(), x)
    dx.Mul(w.T(), dz.T())
    return &dw, &dx
}

func forward_add_gate(x1 *mat.Dense, x2 *mat.Dense) *mat.Dense{
    x1.Add(x1, x2)
    return x1
}

func backward_add_gate(x1 *mat.Dense, x2 *mat.Dense, dz *mat.Dense) (*mat.Dense, *mat.Dense){
    rows, columns := x1.Dims()
    ones_x1 := generateConstant(rows, columns, 1)
    rows2, columns2 := x2.Dims()
    ones_x2 := generateConstant(rows2, columns2, 1)
    ones_x1.MulElem(dz, ones_x1)
    ones_x2.MulElem(dz, ones_x2)
    return ones_x1, ones_x2
}

func forward_recurrent(x *mat.Dense, prev_state *mat.Dense, 
                       U *mat.Dense, W *mat.Dense, V *mat.Dense) (*mat.Dense, *mat.Dense, 
                                                                  *mat.Dense, *mat.Dense, *mat.Dense){
    mul_u := forward_multiply_gate(x, mat.DenseCopyOf(U.T()))
    mul_w := forward_multiply_gate(prev_state, mat.DenseCopyOf(W.T()))
    add_previous_now := forward_add_gate(mul_u, mul_w)
    current_state := calculateTanh(add_previous_now,false)
    mul_v := forward_multiply_gate(current_state, mat.DenseCopyOf(V.T()))
    return mul_u, mul_w, add_previous_now, current_state, mul_v
}

func backward_recurrent(x *mat.Dense, prev_state *mat.Dense, 
                        U *mat.Dense, W *mat.Dense, V *mat.Dense, d_mul_v *mat.Dense,
                        mul_u *mat.Dense, mul_w *mat.Dense, add_previous_now *mat.Dense, 
                        current_state *mat.Dense, mul_v *mat.Dense) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense){
    dV, dcurrent_state := backward_multiply_gate(V, current_state, d_mul_v)
    dadd_previous_now := calculateTanh(add_previous_now, true)
    dadd_previous_now.MulElem(dadd_previous_now, dcurrent_state.T())
    dmul_w, dmul_u := backward_add_gate(mul_w, mul_u, dadd_previous_now)
    dW, dprev_state := backward_multiply_gate(W, prev_state, dmul_w)
    dU, dx := backward_multiply_gate(U, x, dmul_u)
    return dprev_state, dU, dW, dV
}
