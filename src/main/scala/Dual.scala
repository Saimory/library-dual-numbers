package com.example.dualnumber

class Dual(val real: Double, val epsilon: Double) {
  // Операции сложения, вычитания, умножения и деления для дуальных чисел
  def +(other: Dual): Dual = new Dual(real + other.real, epsilon + other.epsilon)
  def -(other: Dual): Dual = new Dual(real - other.real, epsilon - other.epsilon)
  def *(other: Dual): Dual = new Dual(real * other.real, real * other.epsilon + epsilon * other.real)
  def /(other: Dual): Dual = new Dual(real / other.real, (epsilon * other.real - real * other.epsilon) / (other.real * other.real))

  // Перегрузка операторов сравнения
  def >(other: Dual): Boolean = real > other.real
  def <(other: Dual): Boolean = real < other.real
  def ==(other: Dual): Boolean = real == other.real && epsilon == other.epsilon

  // Перегрузка операторов для сравнения с обычными числами
  def >(other: Double): Boolean = real > other
  def <(other: Double): Boolean = real < other
  def ==(other: Double): Boolean = real == other

  // Тригонометрические функции для дуальных чисел
  def sin: Dual = new Dual(Math.sin(real), epsilon * Math.cos(real))
  def cos: Dual = new Dual(Math.cos(real), -epsilon * Math.sin(real))
  def tan: Dual = { new Dual(Math.tan(real), epsilon / (Math.cos(real) * Math.cos(real))) }
  def cot: Dual = { new Dual(1.0 / Math.tan(real), -epsilon / (Math.sin(real) * Math.sin(real))) }

  // Обратные тригонометрические функции для дуальных чисел
  def asin: Dual = { new Dual(Math.asin(real), epsilon / Math.sqrt(1 - real * real)) }
  def acos: Dual = { new Dual(Math.acos(real), -epsilon / Math.sqrt(1 - real * real)) }
  def atan: Dual = { new Dual(Math.atan(real), epsilon / (1 + real * real)) }
  def acot: Dual = { new Dual(Math.atan(1 / real), -epsilon / (1 + real * real)) }

  // Гиперболические функции для дуальных чисел
  def sinh: Dual = { new Dual(Math.sinh(real), epsilon * Math.cosh(real)) }
  def cosh: Dual = { new Dual(Math.cosh(real), epsilon * Math.sinh(real)) }
  def tanh: Dual = { new Dual(Math.tanh(real), epsilon / (Math.cosh(real) * Math.cosh(real))) }
  def coth: Dual = { new Dual(1.0 / Math.tanh(real), -epsilon / (Math.sinh(real) * Math.sinh(real))) }


  // Экспоненциальная и логарифмическая функции для дуальных чисел
  def exp: Dual = new Dual(Math.exp(real), epsilon * Math.exp(real))
  def log: Dual = new Dual(Math.log(real), epsilon / real)
  def logWithBaseAndDual(x: Dual, base: Double): Dual = { Dual(Math.log(real) / Math.log(base), epsilon / (Math.log(base) * real)) }

  // Преобразование дуального числа в обычное число
  def toDouble: Double = real

  // Вычисление производной функции по переменной
  def derivative: Double = epsilon

  // Вычисление корня n-ой степени
  def root(n: Int): Dual = {
    new Dual(scala.math.pow(real, 1.0 / n), epsilon / (n * scala.math.pow(real, (n - 1.0) / n)))
  }

  // Возведение в степень для дуальных чисел
  def pow(exponent: Int): Dual = exponent match {
    case 0 => Dual(1, 0)
    case 1 => this
    case _ => pow(exponent - 1) * this
  }
  // Возведение в степень для дуальных чисел с помощью другого дуального числа
  def pow(other: Dual): Dual = {
    if (other.real == 0) Dual(1, 0)
    else if (other.real == 1) this
    else {
      val powResult = pow(other.real)
      val powReal = powResult.real
      val powEpsilon = powReal * (other.epsilon * Math.log(real) + other.real * epsilon / real)
      new Dual(powReal, powEpsilon)
    }
  }

  // Возведение вещественного числа в дуальную степень
  def pow(realExponent: Double): Dual = {
    val powReal = Math.pow(real, realExponent)
    val powEpsilon = realExponent * powReal * epsilon / real
    new Dual(powReal, powEpsilon)
  }

  def printMatrix(matrix: Dual): Unit = {
    println("Матричное представление дуального числа:")
    println(f"[ ${matrix.real}%.2f\t${matrix.epsilon}%.2f ]")
    println(f"[ 0.00\t${matrix.real}%.2f ]")
  }

  // Переопределение метода toString для красивого вывода
  override def toString: String = s"$real + $epsilon e"

}


object Dual {
  // Метод для создания дуального числа
  def apply(real: Double, epsilon: Double): Dual = new Dual(real, epsilon)
  def variable(name: String): Dual = new Dual(0, 1)
  // Метод для создания дуального числа из обычного числа (с нулевой мнимой частью)
  def toDual(value: Double): Dual = Dual(value, 0)
}
