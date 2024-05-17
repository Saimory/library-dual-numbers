import com.example.dualnumber.Dual
import scala.math._

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
