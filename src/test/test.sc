import com.example.dualnumber.Dual
import scala.math._
import scala.util.Random

// Создание дуального числа
val dualNumber = Dual(3.0, 2.0)
println("Дуальное число: " + dualNumber)

// Примеры операций
val sum = dualNumber + Dual(1.0, 1.0)
val difference = dualNumber - Dual(1.0, 1.0)
val product = dualNumber * Dual(2.0, 1.0)
val quotient = dualNumber / Dual(2.0, 1.0)
println("Сумма: " + sum)
println("Разность: " + difference)
println("Произведение: " + product)
println("Частное: " + quotient)
// Примеры сравнения
val more = dualNumber < Dual(4.0, 1.0)
val less = dualNumber > 5.0
println("Сравнение дуальных чисел Dual(3.0, 2.0) < Dual(4.0, 1.0): " + more)
println("Сравнение дуального и обычного числа Dual(3.0, 2.0) > 5.0: " + less)

// Примеры тригонометрических функций
val sine = dualNumber.sin
val cosine = dualNumber.cos
val tangent = dualNumber.tan
val cotangent = dualNumber.cot
println("Синус: " + sine)
println("Косинус: " + cosine)
println("Тангенс: " + tangent)
println("Котангенс: " + cotangent)

// Примеры обратных тригонометрических функций
val arcsine = dualNumber.asin
val arccosine = dualNumber.acos
val arctangent = dualNumber.atan
val arccotangent = dualNumber.acot
println("Арксинус: " + arcsine)
println("Арккосинус: " + arccosine)
println("Арктангенс: " + arctangent)
println("Арккотангенс: " + arccotangent)

// Примеры гиперболических функций
val hyperbolicSine = dualNumber.sinh
val hyperbolicCosine = dualNumber.cosh
val hyperbolicTangent = dualNumber.tanh
val hyperbolicCotangent = dualNumber.coth
println("Гиперболический синус: " + hyperbolicSine)
println("Гиперболический косинус: " + hyperbolicCosine)
println("Гиперболический тангенс: " + hyperbolicTangent)
println("Гиперболический котангенс: " + hyperbolicCotangent)

// Примеры экспоненциальной и логарифмической функций
val exponent = dualNumber.exp
val logarithm = dualNumber.log
println("Экспонента: " + exponent)
println("Логарифм: " + logarithm)
val x = Dual(4, 1)
val result = x.logWithBaseAndDual(Dual(4, 1), 2) // log_2(4)
println(result) // Выведет приблизительное значение log_2(4) и его производной

// Примеры возведения в степень и вычисления корня
val power = dualNumber.pow(3)
val power_dual = dualNumber.pow(dualNumber)
val root = dualNumber.root(2)
println("Возведение дуального числа в вещественную степень: " + power)
println("Возведение дуального числа в дуальную степень: " + power_dual)
println("Корень: " + root)

// Пример преобразования дуального числа в вещественное число
val toDouble = dualNumber.toDouble
println("Преобразование в обычное число: " + toDouble)

val matrix = dualNumber.printMatrix(dualNumber)








// Для примера использования библиотеки дифференцируем функцию

// Обычный способ
def f(z: Double): Double = { 3 + z * log(z * z) }
def df(z: Double): Double = { log(z*z) + 2 }

val z = 5.0
val result = f(z)
println("Результат функции f(z): " + result)
val result = df(z)
println("Результат функции df(z): " + result)


// Константу 3 представляем в виде 3 = 3 + 0 * e
val const3 = Dual.toDual(3)
// Вычисление производим в точке z = 5 + 1 * e
val result = const3 + Dual(5, 1) * Dual(5, 1).pow(2).log

println("Значение функции f(x) в точке x = 5: " + result.real)
println("Значение производной f'(x) в точке x = 5: " + result.derivative)

// Создание и обучение нейронной сети для аппроксимации функции синуса на заданном интервале.
class Neuron(val inputWeights: Array[Dual]) {
  // Метод активации нейрона.
  // Принимает массив входных данных (inputs) и возвращает результат активации нейрона.
  def activate(inputs: Array[Dual]): Dual = {
    // Активирует нейрон, принимая входные данные и взвешенные коэффициенты,
    // считает взвешенную сумму и передает ее в сигмоидную функцию активации
    val weightedSum = inputWeights.zip(inputs).map { case (w, i) => w * i }.reduce(_ + _)
    sigmoid(weightedSum)
  }
  // Сигмоидная функция активации.
  // Принимает значение x и возвращает результат применения сигмоидной функции к нему.
  def sigmoid(x: Dual): Dual = {
    // Вычисляет сигмоидную функцию для заданного значения
    val expValue = (-x).exp
    Dual(1.0, 0.0) / (Dual(1.0, 0.0) + expValue)
  }
  // Производная сигмоидной функции активации.
  // Принимает выходной сигнал (output) и возвращает производную сигмоидной функции для данного выхода.
  def sigmoidDerivative(output: Dual): Dual = output * (Dual(1.0, 0.0) - output)
}

class NeuralNetwork(val neurons: Array[Neuron]) {
  def predict(inputs: Array[Dual]): Array[Dual] = {
    // Предсказывает выходные значения нейронной сети для заданных входных данных
    neurons.map(_.activate(inputs))
  }

  def train(data: Array[(Array[Double], Double)], epochs: Int, learningRate: Double): Unit = {
    // Обучает нейронную сеть на основе предоставленных данных
    for (epoch <- 1 to epochs) {
      data.foreach { case (inputs, target) =>
        val dualInputs = inputs.map(Dual(_, 0.0))
        // Предсказывает выходы для текущих входных данных.
        val outputs = predict(dualInputs)
        // Ошибка выхода
        val outputError = Dual(target, 0.0) - outputs.head
        // Дельта выхода для корректировки весов
        val outputDelta = outputError * neurons.head.sigmoidDerivative(outputs.head)
        //  Обновление весов
        neurons.head.inputWeights.indices.foreach { i =>
          val inputDerivative = Dual(dualInputs(i).real, dualInputs(i).epsilon)
          neurons.head.inputWeights(i) = neurons.head.inputWeights(i) + Dual(learningRate, 0.0) * outputDelta * inputDerivative
        }
      }
    }
  }
}

// Подготовка данных для обучения
val trainingData: Array[(Array[Double], Double)] = (for {
  x <- Range.BigDecimal(-Math.PI, Math.PI, BigDecimal(0.1)).map(_.toDouble)
} yield (Array(x), Math.sin(x))).toArray

// Инициализация весов и сети
val initialWeights: Array[Dual] = Array(Dual(scala.util.Random.nextDouble(), 0.0))
val neuron = new Neuron(initialWeights)
val neuralNetwork = new NeuralNetwork(Array(neuron))

// Обучение сети
neuralNetwork.train(trainingData, epochs = 10, learningRate = 0.01)

// Тестирование сети на нескольких значениях
val testValues = Array(Math.PI / 4, Math.PI / 2, Math.PI / 6, Math.PI / 3)
val correctValues = testValues.map(Math.sin)
val testInputs = testValues.map(x => Array(Dual(x, 1.0)))
val predictions = testInputs.map(neuralNetwork.predict(_).head.toDouble)

// Вывод результатов
testValues.zip(predictions).zip(correctValues).foreach {
  case ((x, prediction), correctValue) =>
    println(f"Предсказанное значение для sin($x%.2f): $prediction%.5f, Правильное значение: $correctValue%.5f")
}





