import com.example.dualnumber.Dual
import scala.math._
import scala.util.Random
import scala.annotation.tailrec

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
println("Значение производной f'(x) в точке x = 5: " + result)
/*
// Создание и обучение нейронной сети для аппроксимации функции синуса на заданном интервале.
class Neuron(var inputWeights: Array[Dual]) {
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
    val expValue = (-x).exp
    Dual(1.0, 0.0) / (Dual(1.0, 0.0) + expValue)
  }
  // Производная сигмоидной функции активации.
  // Принимает выходной сигнал (output) и возвращает производную сигмоидной функции для данного выхода.
  def sigmoidDerivative(output: Dual): Dual = output * (Dual(1.0, 0.0) - output)
}

class NeuralNetwork(val neurons: Array[Neuron]) {
  def predict(inputs: Array[Dual]): Array[Dual] = neurons.map(_.activate(inputs))

  def train(data: Array[(Array[Double], Double)], epochs: Int, learningRate: Double): Unit = {
    // Обучает нейронную сеть на основе предоставленных данных
    for (epoch <- 1 to epochs) {
      data.foreach { case (inputs, target) =>
        val dualInputs = inputs.map(x => Dual(x, 1.0))  // Используем ε = 1.0 для вычисления производных
        // Предсказывает выходы для текущих входных данных.
        val outputs = predict(dualInputs)
        // Ошибка выхода
        val outputError = Dual(target, 0.0) - outputs.head
        // Дельта выхода для корректировки весов
        val outputDelta = outputError * neurons.head.sigmoidDerivative(outputs.head)
        neurons.head.inputWeights.indices.foreach { i =>
          val inputDerivative = dualInputs(i) // Используем дуальные входы с ε = 1.0
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

// Тестирование сети
val testValues = Array(Math.PI / 4, Math.PI / 2, Math.PI / 6, Math.PI / 3)
val correctValues = testValues.map(Math.sin)
val testInputs = testValues.map(x => Array(Dual(x, 0.0))) // ε = 0.0 для тестирования
val predictions = testInputs.map(neuralNetwork.predict(_).head.real)

// Вывод результатов
testValues.zip(predictions).zip(correctValues).foreach {
  case ((x, prediction), correctValue) =>
    println(f"Предсказанное значение для sin($x%.2f): $prediction%.5f, Правильное значение: $correctValue%.5f")
}
 */


// Метод Ньютона для нахождения корня уравнения

def newtonMethod(f: Dual => Dual, tolerance: Double = 1e-7, maxIterations: Int = 100): Double => Double = {
  @tailrec
  def iterate(x: Double, iteration: Int): Double = {
    val fx = f(Dual.variable(x))
    if (Math.abs(fx.real) < tolerance || iteration >= maxIterations) x
    else iterate(x - fx.real / fx.derivative, iteration + 1)
  }
  initialGuess => iterate(initialGuess, 0)
}

// Пример использования:
val equation: Dual => Dual = z => z * z - Dual.toDual(2) // Решаем уравнение z^2 - 2 = 0
val root = newtonMethod(equation)(1.0)
println(s"Корень уравнения z^2 - 2 = 0: $root")


// Пример использования:
def equation(z: Dual): Dual = z * z - Dual.toDual(2) // Решаем уравнение z^2 - 2 = 0
val root = newtonMethod(equation, 1.0)
println(s"Корень уравнения z^2 - 2 = 0: $root")




// Численное интегрирование с чувствительностью к параметрам

def trapezoidalRuleWithDuals(f: Dual => Dual, a: Dual, b: Dual, n: Int): Dual = {
  val h = (b.real - a.real) / n
  val sum = (1 until n).map(i => f(Dual.variable(a.real + i * h)).real).sum
  Dual(0.5 * (f(a).real + f(b).real) + sum * h, 0.5 * (f(a).derivative + f(b).derivative) + sum * h)
}

// Пример использования:
val integrand: Dual => Dual = z => z * z
val integral = trapezoidalRuleWithDuals(integrand, Dual.variable(0.0), Dual.variable(1.0), 1000)
println(s"Численное интегрирование функции z^2 на интервале [0, 1]: ${integral.real}")
println(s"Производная интеграла по начальному значению: ${integral.derivative}")


// Градиентный спуск с использованием дуальных чисел

def gradientDescent(f: Dual => Dual, learningRate: Double = 0.01, tolerance: Double = 1e-7, maxIterations: Int = 1000): Double => Double = {
  @tailrec
  def iterate(x: Double, iteration: Int): Double = {
    val fx = f(Dual.variable(x))
    if (Math.abs(fx.derivative) < tolerance || iteration >= maxIterations) x
    else iterate(x - learningRate * fx.derivative, iteration + 1)
  }
  initialGuess => iterate(initialGuess, 0)
}

// Пример использования:
val costFunction: Dual => Dual = z => (z - Dual.toDual(3)) * (z - Dual.toDual(3))
val minimum = gradientDescent(costFunction)(0.0)
println(s"Минимум функции (z - 3)^2: $minimum")


// Численное решение обыкновенных дифференциальных уравнений (ОДУ) метод Рунге-Кутты 4 порядка
def rk4Method(f: (Dual, Dual) => Dual, y0: Dual, x0: Dual, xEnd: Dual, stepSize: Double): Dual = {
  var x = x0
  var y = y0
  while (x.real < xEnd.real) {
    val k1 = f(x, y)
    val k2 = f(x + Dual.toDual(stepSize / 2), y + k1 * Dual.toDual(stepSize / 2))
    val k3 = f(x + Dual.toDual(stepSize / 2), y + k2 * Dual.toDual(stepSize / 2))
    val k4 = f(x + Dual.toDual(stepSize), y + k3 * Dual.toDual(stepSize))
    y = y + (k1 + k2 * Dual.toDual(2) + k3 * Dual.toDual(2) + k4) * Dual.toDual(stepSize / 6)
    x = x + Dual.toDual(stepSize)
  }
  y
}

// Пример использования
def diffEq(x: Dual, y: Dual): Dual = x + y
val y0Dual = Dual(1.0, 1.0)  // y0 = 1.0, с производной 1.0
val x0Dual = Dual(0.0, 0.0)  // x0 = 0.0, с производной 0.0
val xEndDual = Dual(1.0, 0.0) // xEnd = 1.0, с производной 0.0
val stepSize = 0.01
val solutionDual = rk4Method(diffEq, y0Dual, x0Dual, xEndDual, stepSize)
println(s"Решение дифференциального уравнения dy/dx = x + y при y(0) = 1 методом Рунге-Кутты: $solutionDual")
println(s"Значение производной решения при x = 1: ${solutionDual.derivative}")

/*
Предположим, что у нас есть автомобиль, движущийся по траектории,
описанной функцией положения x(t), где t — время. Нам необходимо
оптимизировать траекторию движения автомобиля таким образом,
чтобы минимизировать его суммарную энергию, затраченную на движение.
Функция энергии задается как интеграл от квадрата скорости
 */

object CarEnergyOptimization {
  val T = 10.0 // Конечное время

  // Функция положения
  def position(t: Dual): Dual = {
    Dual.constant(10) * t.sin + Dual.constant(5) * t
  }
  // Энергия вычисляется как квадрат скорости
  def energy(v: Dual): Dual = {
    v * v
  }

  def integrateEnergy(dt: Double): Double = {
    var totalEnergy = 0.0
    var t = 5.0
    while (t <= T) {
      val dualT = Dual.variable(t)
      val pos = position(dualT)
      val vel = pos.epsilon
      val e = energy(Dual(vel, 0))
      totalEnergy += e.real * dt
      t += dt
    }
    totalEnergy
  }

  def main(args: Array[String]): Unit = {
    val dt = 0.01
    val totalEnergy = integrateEnergy(dt)
    println(f"Суммарная энергия: $totalEnergy%.4f")
  }
}

CarEnergyOptimization.main(Array())

