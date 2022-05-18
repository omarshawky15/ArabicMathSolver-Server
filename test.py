import unittest

from app import predict
from evaluation import polynomial
from translation import translate_to_arabic_html


def get_Prediction(img_path):
    expression, mapping = predict(img_path)
    solution, error = polynomial(expression, mapping)
    arabic_expr, arabic_sol = translate_to_arabic_html(expression, solution, mapping)
    prediction = {'expression': expression, 'mapping': str(mapping), 'solution': str(solution), 'error': str(error),
                  'arabic_expr': arabic_expr, 'arabic_sol': arabic_sol}
    return prediction


class MyTestCase(unittest.TestCase):

    def test_pow(self):
        prediction = get_Prediction('test_images/IMG_20220406_164831_24.jpg')
        self.assertEqual(prediction['expression'], 'x^(3*x+5)')
        self.assertEqual(prediction['solution'], '[0]')
        self.assertEqual(prediction['arabic_expr'], '<html><body>س <sup>٣*س+٥</sup></body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>٠</body></html>'])

        prediction = get_Prediction('test_images/IMG_20220517_125215_763.png')
        self.assertEqual(prediction['expression'], '(x+3)^5')
        self.assertEqual(prediction['solution'], '[-3]')
        self.assertEqual(prediction['arabic_expr'], '<html><body>(ب+٣) <sup>٥</sup></body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body> -٣</body></html>'])

        prediction = get_Prediction('test_images/IMG_5375.png')
        self.assertEqual(prediction['expression'], '10^(12)')
        self.assertEqual(prediction['solution'], '1000000000000.00')
        self.assertEqual(prediction['arabic_expr'], '<html><body>١٠ <sup>١٢</sup></body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>١٠٠٠٠٠٠٠٠٠٠٠٠,٠٠</body></html>'])

        prediction = get_Prediction('test_images/IMG_2.png')
        self.assertEqual(prediction['expression'], '2^x-3')
        self.assertEqual(prediction['solution'], '[log(3)/log(2)]')
        self.assertEqual(prediction['arabic_expr'], '<html><body>٢ <sup>س</sup> -٣</body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>لـو(٣)\\لـو(٢)</body></html>'])

        prediction = get_Prediction('test_images/IMG_4.png')
        self.assertEqual(prediction['expression'], '2^3-5')
        self.assertEqual(prediction['solution'], '3.00000000000000')
        self.assertEqual(prediction['arabic_expr'], '<html><body>٢ <sup>٣</sup> -٥</body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>٣,٠٠٠٠٠٠٠٠٠٠٠٠٠٠</body></html>'])

    def test_fraction(self):
        prediction = get_Prediction('test_images/IMG_9.png')
        self.assertEqual(prediction['expression'], '(x+1/2)/(x^2+3/4)=4/5')
        self.assertEqual(prediction['solution'], '[5/8 - sqrt(17)/8, sqrt(17)/8 + 5/8]')
        self.assertEqual(prediction['arabic_expr'],
                         '<html><body>(س+١\\٢)\\(س <sup>٢</sup>+٣\\٤)=٤\\٥</body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>٥\\٨ -√(١٧)\\٨</body></html>',
                                                    '<html><body>√(١٧)\\٨+٥\\٨</body></html>'])

    def test_sqrt(self):
        prediction = get_Prediction('test_images/IMG_20220325_180702_831.jpg')
        self.assertEqual(prediction['expression'], 'sqrt(x^2+2*x+4)=0')
        self.assertEqual(prediction['solution'], '[-1 - sqrt(3)*I, -1 + sqrt(3)*I]')
        self.assertEqual(prediction['arabic_expr'], '<html><body>√(س <sup>٢</sup>+٢*س+٤)=٠</body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body> -١ -√(٣)*ت</body></html>',
                                                    '<html><body> -١+√(٣)*ت</body></html>'])

    def test_decimal_point(self):
        prediction = get_Prediction('test_images/IMG_20220325_183755_522.jpg')
        self.assertEqual(prediction['expression'], '(5.3+7)/5+1')
        self.assertEqual(prediction['solution'], '3.46000000000000')

    def test_times(self):
        prediction = get_Prediction('test_images/IMG_10.png')
        self.assertEqual(prediction['expression'], '2*x*sin(x^2)')
        self.assertEqual(prediction['solution'], '[0, -sqrt(pi), sqrt(pi)]')
        self.assertEqual(prediction['arabic_expr'], '<html><body>٢*س*جا(س <sup>٢</sup>)</body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>٠</body></html>', '<html><body> -√(ط)</body></html>',
                                                    '<html><body>√(ط)</body></html>'])

        prediction = get_Prediction('test_images/IMG_11.png')
        self.assertEqual(prediction['expression'], '(1+1)*(2+3)')
        self.assertEqual(prediction['solution'], '10.0000000000000')
        self.assertEqual(prediction['arabic_expr'], '<html><body>(١+١)*(٢+٣)</body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>١٠,٠٠٠٠٠٠٠٠٠٠٠٠٠</body></html>'])

    def test_trig_functions(self):
        prediction = get_Prediction('test_images/IMG_20220330_185200_343.jpg')
        self.assertEqual(prediction['expression'], 'sin(2*pi)+1')
        self.assertEqual(prediction['solution'], '1.00000000000000')

        prediction = get_Prediction('test_images/IMG_20220406_175855_152.jpg')
        self.assertEqual(prediction['expression'], 'tan(pi*E)')
        self.assertEqual(prediction['solution'], '-1.22216467181901')

        prediction = get_Prediction('test_images/IMG_20220406_190500_200.jpg')
        self.assertEqual(prediction['expression'], 'sin(6*pi+3)')
        self.assertEqual(prediction['solution'], '0.141120008059867')

        prediction = get_Prediction('test_images/IMG_20220406_212851_465.jpg')
        self.assertEqual(prediction['expression'], 'sec(6*pi+3)')
        self.assertEqual(prediction['solution'], '-1.01010866590799')

        prediction = get_Prediction('test_images/IMG_20220406_222734_822.jpg')
        self.assertEqual(prediction['expression'], 'sin(E)^2+cos(E)^2')
        self.assertEqual(prediction['solution'], '1.00000000000000')
        self.assertEqual(prediction['arabic_expr'],
                         '<html><body>جا(هـ) <sup>٢</sup>+جتا(هـ) <sup>٢</sup></body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>١,٠٠٠٠٠٠٠٠٠٠٠٠٠٠</body></html>'])

        prediction = get_Prediction('test_images/IMG_20220517_141952_471.jpg')
        self.assertEqual(prediction['expression'], '(sec(3*pi))/(csc(3*pi))')
        self.assertEqual(prediction['solution'], '0')
        self.assertEqual(prediction['arabic_expr'], '<html><body>(قا(٣*ط))\\(قتا(٣*ط))</body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>٠</body></html>'])

    def test_poly_equations(self):
        prediction = get_Prediction('test_images/IMG_20220404_001419_252.jpg')
        self.assertEqual(prediction['expression'], 'x^2+3*x=-2')
        self.assertEqual(prediction['solution'], '[-2, -1]')

        prediction = get_Prediction('test_images/IMG_20220404_003309_121.jpg')
        self.assertEqual(prediction['expression'], 'x^2+3*x+2')
        self.assertEqual(prediction['solution'], '[-2, -1]')
        self.assertEqual(prediction['arabic_expr'], '<html><body>س <sup>٢</sup>+٣*س+٢</body></html>')
        self.assertEqual(prediction['arabic_sol'],
                         ['<html><body> -٢</body></html>', '<html><body> -١</body></html>'])

        prediction = get_Prediction('test_images/IMG_1.png')
        self.assertEqual(prediction['expression'], 'x^3-x^2-7')
        self.assertEqual(prediction['solution'],
                         '[1/3 + (-1/2 - sqrt(3)*I/2)*(sqrt(4053)/18 + 191/54)**(1/3) + 1/(9*(-1/2 - sqrt(3)*I/2)*('
                         'sqrt(4053)/18 + 191/54)**(1/3)), 1/3 + 1/(9*(-1/2 + sqrt(3)*I/2)*(sqrt(4053)/18 + '
                         '191/54)**(1/3)) + (-1/2 + sqrt(3)*I/2)*(sqrt(4053)/18 + 191/54)**(1/3), 1/(9*(sqrt(4053)/18 '
                         '+ 191/54)**(1/3)) + 1/3 + (sqrt(4053)/18 + 191/54)**(1/3)]')
        self.assertEqual(prediction['arabic_expr'],
                         '<html><body>س <sup>٣</sup> -س <sup>٢</sup> -٧</body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>١\\٣+( -١\\٢ -√(٣)*ت\\٢)*(√(٤٠٥٣)\\١٨+١٩١\\٥٤) <sup>١\\٣</sup>+١\\(٩*( -١\\٢ -√(٣)*ت\\٢)*(√(٤٠٥٣)\\١٨+١٩١\\٥٤) <sup>١\\٣</sup>)</body></html>',
                                                    '<html><body>١\\٣+١\\(٩*( -١\\٢+√(٣)*ت\\٢)*(√(٤٠٥٣)\\١٨+١٩١\\٥٤) <sup>١\\٣</sup>)+( -١\\٢+√(٣)*ت\\٢)*(√(٤٠٥٣)\\١٨+١٩١\\٥٤) <sup>١\\٣</sup></body></html>',
                                                    '<html><body>١\\(٩*(√(٤٠٥٣)\\١٨+١٩١\\٥٤) <sup>١\\٣</sup>)+١\\٣+(√(٤٠٥٣)\\١٨+١٩١\\٥٤) <sup>١\\٣</sup></body></html>'])

        prediction = get_Prediction('test_images/IMG_3.png')
        self.assertEqual(prediction['expression'], 'x^2-4=0')
        self.assertEqual(prediction['solution'], '[-2, 2]')
        self.assertEqual(prediction['arabic_expr'], '<html><body>س <sup>٢</sup> -٤=٠</body></html>')
        self.assertEqual(prediction['arabic_sol'],
                         ['<html><body> -٢</body></html>', '<html><body>٢</body></html>'])

        prediction = get_Prediction('test_images/IMG_6.png')
        self.assertEqual(prediction['expression'], 'x^2-9=0')
        self.assertEqual(prediction['solution'], '[-3, 3]')
        self.assertEqual(prediction['arabic_expr'], '<html><body>س <sup>٢</sup> -٩=٠</body></html>')
        self.assertEqual(prediction['arabic_sol'],
                         ['<html><body> -٣</body></html>', '<html><body>٣</body></html>'])

    def test_log(self):
        prediction = get_Prediction('test_images/IMG_20220406_185940_704.jpg')
        self.assertEqual(prediction['expression'], 'log(pi,pi)')
        self.assertEqual(prediction['solution'], '1.00000000000000')

        prediction = get_Prediction('test_images/IMG_5.png')
        self.assertEqual(prediction['expression'], 'log(2*2-3,10)')
        self.assertEqual(prediction['solution'], '0')
        self.assertEqual(prediction['arabic_expr'], '<html><body>لـو<sub>١٠</sub>(٢*٢ -٣)</body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>٠</body></html>'])

    def test_complex_expr(self):
        prediction = get_Prediction('test_images/IMG_20220515_205442_998.jpg')
        self.assertEqual(prediction['expression'], '(-x+sqrt(x^2+4*x*pi))/(sin(pi)+log(10,10))')
        self.assertEqual(prediction['solution'], '[0]')

    def test_more_than_1_variable(self):
        prediction = get_Prediction('test_images/IMG_7.png')
        self.assertEqual(prediction['expression'], 'x^2+y^2=0')
        self.assertEqual(prediction['solution'], '[(-I*y, y), (I*y, y)]')
        self.assertEqual(prediction['arabic_expr'], '<html><body>س <sup>٢</sup>+ص <sup>٢</sup>=٠</body></html>')
        self.assertEqual(prediction['arabic_sol'],
                         ['<html><body>( -ت*ص,ص)</body></html>', '<html><body>(ت*ص,ص)</body></html>'])

        prediction = get_Prediction('test_images/IMG_8.png')
        self.assertEqual(prediction['expression'], 'x+y-z+a')
        self.assertEqual(prediction['solution'], '[(-a - y + z, y, z, a)]')
        self.assertEqual(prediction['arabic_expr'], '<html><body>أ+ب -جـ+د</body></html>')
        self.assertEqual(prediction['arabic_sol'], ['<html><body>( -د -ب+جـ,ب,جـ,د)</body></html>'])


if __name__ == '__main__':
    unittest.main()
