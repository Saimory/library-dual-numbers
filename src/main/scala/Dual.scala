package com.example.dualnumber

class Dual(val real: Double, val epsilon: Double) {
  def +(other: Dual): Dual = new Dual(real + other.real, epsilon + other.epsilon)
  def -(other: Dual): Dual = new Dual(real - other.real, epsilon - other.epsilon)
  def *(other: Dual): Dual = new Dual(real * other.real, real * other.epsilon + epsilon * other.real)
  def /(other: Dual): Dual = new Dual(real / other.real, (epsilon * other.real - real * other.epsilon) / (other.real * other.real))

  def >(other: Dual): Boolean = real > other.real
  def <(other: Dual): Boolean = real < other.real
  def ==(other: Dual): Boolean = real == other.real && epsilon == other.epsilon
  def unary_- : Dual = new Dual(-real, -epsilon)

  def >(other: Double): Boolean = real > other
  def <(other: Double): Boolean = real < other
  def ==(other: Double): Boolean = real == other

  def sin: Dual = new Dual(Math.sin(real), epsilon * Math.cos(real))
  def cos: Dual = new Dual(Math.cos(real), -epsilon * Math.sin(real))
  def tan: Dual = new Dual(Math.tan(real), epsilon / (Math.cos(real) * Math.cos(real)))
  def cot: Dual = new Dual(1.0 / Math.tan(real), -epsilon / (Math.sin(real) * Math.sin(real)))

  def asin: Dual = new Dual(Math.asin(real), epsilon / Math.sqrt(1 - real * real))
  def acos: Dual = new Dual(Math.acos(real), -epsilon / Math.sqrt(1 - real * real))
  def atan: Dual = new Dual(Math.atan(real), epsilon / (1 + real * real))
  def acot: Dual = new Dual(Math.atan(1 / real), -epsilon / (1 + real * real))

  def sinh: Dual = new Dual(Math.sinh(real), epsilon * Math.cosh(real))
  def cosh: Dual = new Dual(Math.cosh(real), epsilon * Math.sinh(real))
  def tanh: Dual = new Dual(Math.tanh(real), epsilon / (Math.cosh(real) * Math.cosh(real)))
  def coth: Dual = new Dual(1.0 / Math.tanh(real), -epsilon / (Math.sinh(real) * Math.sinh(real)))

  def exp: Dual = new Dual(Math.exp(real), epsilon * Math.exp(real))
  def log: Dual = new Dual(Math.log(real), epsilon / real)
  def logWithBaseAndDual(x: Dual, base: Double): Dual = Dual(Math.log(real) / Math.log(base), epsilon / (Math.log(base) * real))

  def toDouble: Double = real
  def derivative: Double = epsilon
  def root(n: Int): Dual = new Dual(scala.math.pow(real, 1.0 / n), epsilon / (n * scala.math.pow(real, (n - 1.0) / n)))

  def printMatrix(matrix: Dual): Unit = {
    println("Матричное представление дуального числа:")
    println(f"[ ${matrix.real}%.2f\t${matrix.epsilon}%.2f ]")
    println(f"[ 0.00\t${matrix.real}%.2f ]")
  }

  def pow(exponent: Int): Dual = exponent match {
    case 0 => Dual(1, 0)
    case 1 => this
    case _ => pow(exponent - 1) * this
  }
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

  def pow(realExponent: Double): Dual = {
    val powReal = Math.pow(real, realExponent)
    val powEpsilon = realExponent * powReal * epsilon / real
    new Dual(powReal, powEpsilon)
  }

  override def toString: String = s"$real + $epsilon e"
}

object Dual {
  def apply(real: Double, epsilon: Double): Dual = new Dual(real, epsilon)
  def variable(name: String): Dual = new Dual(0, 1)
  def toDual(value: Double): Dual = Dual(value, 0)
  def variable(real: Double): Dual = new Dual(real, 1)
  def constant(real: Double): Dual = new Dual(real, 0)
}

