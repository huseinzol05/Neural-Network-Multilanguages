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

// send address, printMatrix(&X)
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

// generate ones, ones := generateConstant(150, 3, 1)
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

const EPOCH = 50
const LEARNING_RATE = 0.0001

func main() {
    file, _ := os.Open("iris.csv")
    defer file.Close()
    reader := csv.NewReader(file)
    record, _ := reader.ReadAll()
    record = record[1:]
    row_size := len(record)
    column_size := len(record[0][1:len(record[0])-1])
    iris := make([]float64, row_size * column_size)
    flowers := make([]string, row_size)
    
    for i := 0; i < row_size; i++ {
        for k := 0; k < column_size; k++ {
            val, _ := strconv.ParseFloat(record[i][k+1],64)
            iris[(i*column_size)+k] = val
        }
        flowers[i] = record[i][len(record[i])-1]
    }
    x_iris := mat.NewDense(row_size, column_size, iris)
    unique_flowers := removeDuplicates(flowers)
    
    // code our onehot-encoder
    onehot := mat.NewDense(row_size, len(unique_flowers), nil)
    for i := 0; i < row_size; i++ {
        onehot.Set(i, indexOf(flowers[i], unique_flowers), 1)
    }
    
    W1 := randomMatrix(column_size, 64)
    W2 := randomMatrix(64, 64)
    W3 := randomMatrix(64, len(unique_flowers))

    applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v,false) }
    applySigmoidDerivative := func(_, _ int, v float64) float64 { return sigmoid(v,true) }
    for iteration := 0; iteration < EPOCH; iteration++ {
        var a1 mat.Dense
        a1.Mul(x_iris, W1)
        z1 := new(mat.Dense)
        z1.Apply(applySigmoid, &a1)
        var a2 mat.Dense
        a2.Mul(z1, W2)
        z2 := new(mat.Dense)
        z2.Apply(applySigmoid, &a2)
        var a3 mat.Dense
        a3.Mul(z2, W3)
        y_hat := softmax(&a3,false)
        accuracy := getAccuracy(y_hat, onehot)
        err := calculateCrossEntropy(y_hat, onehot,false)
        cost := mat.Sum(err) / float64(column_size * row_size)
        fmt.Printf("epoch %d, cost %f, accuracy %f\n", iteration, cost, accuracy)
        dy_hat := calculateCrossEntropy(y_hat, onehot,true)
        da3 := softmax(&a3,true)
        da3.MulElem(da3, dy_hat)
        var dW3 mat.Dense
        dW3.Mul(z2.T(), da3)
        var dz2 mat.Dense
        dz2.Mul(da3, W3.T())
        da2 := new(mat.Dense)
        da2.Apply(applySigmoidDerivative, &a2)
        da2.MulElem(da2, &dz2)
        var dW2 mat.Dense
        dW2.Mul(z1.T(), da2)
        var dz1 mat.Dense
        dz1.Mul(da2, W2.T())
        da1 := new(mat.Dense)
        da1.Apply(applySigmoidDerivative, &a1)
        da1.MulElem(da1, &dz1)
        var dW1 mat.Dense
        dW1.Mul(x_iris.T(), da1)
        dW3.Scale(-LEARNING_RATE, &dW3)
        dW2.Scale(-LEARNING_RATE, &dW2)
        dW1.Scale(-LEARNING_RATE, &dW1)
        W3.Add(W3,&dW3)
        W2.Add(W2,&dW2)
        W1.Add(W1,&dW1)
    }
}