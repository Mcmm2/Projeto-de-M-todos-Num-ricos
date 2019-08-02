import matplotlib.pyplot as plt
import sympy as sp

class ODE:
    """Possui vários métodos para resolver a EDO y'(t) = f(t, y(t)); y(t0) = y0"""

    adam = {1: [1],
            2: [3/2, -1/2],
            3: [22/12, -4/3, 5/12],
            4: [55/24, -59/24, 37/24, -3/8],
            5: [1901/720, -1387/360, 109/30, -637/360, 251/720],
            6: [4277/1440, -2641/480, 4991/720, -3649/720, 959/480, -95/288],
            7: [198721/60480, -18367/2520, 235183/20160, -10754/945, 135713/20160, -5603/2520, 19087/60480],
            8: [16083/4480, -1152169/120960, 242653/13440, -296053/13440, 2102243/120960, -115747/13440, 32863/13440, -5257/17280]
             }
    multon = {1: [1],
              2: [1/2, 1/2],
              3: [5/12, 2/3, -1/12],
              4: [3/8, 19/24, -5/24, 1/24],
              5: [251/720, 323/360, -11/30, 53/360, -19/720],
              6: [95/288, 1427/1440, -133/240, 241/720, -173/1440, 3/160],
              7: [19087/60480, 2713/2520, -15487/20160, 586/945, -6737/20160, 263/2520, -863/60480],
              8: [5257/17280, 139849/120960, -4511/4480, 123133/120960, -88547/120960, 1537/4480, -11351/120960, 275/24192]
               }

    diff = {1: [1, 1],
            2: [2/3, 4/3, -1/3],
            3: [6/11, 18/11, -9/11, 2/11],
            4: [12/25, 48/25, -36/25, 16/25, -3/25],
            5: [60/137, 300/137, -300/137, 200/137, -75/137, 12/137],
            6: [60/147, 360/147, -450/147, 400/147, -225/147, 72/147, -10/147]	
             }

    def __init__(self, t0, y0, str_expr):
        """Valor inicial(t0, y0) e expressão(f(t, y) """

        t, y = sp.symbols('t, y')
        expr = sp.sympify(str_expr)
        self.t0 = t0
        self.y0 = y0
        self.f = sp.lambdify((t, y), expr)
     
        return None

    def coef_adam(self, order):
        """Coeficientes de Adams Bashforth de qualquer ordem"""

        x = sp.symbols('x')
        if order not in self.adam:
            self.adam[order] = []
            for i in reversed(range(order)):
                str_expr = '1'
                for j in range(order):
                    if j == order - i - 1:
                        continue
                    str_expr += f' * (x + {j})'
                expr = sp.sympify(str_expr)
                expr = sp.lambdify((x), expr)
                value = sp.integrate(expr(x), (x, 0, 1))
                self.adam[order].append(((-1)**(order - i - 1) * value) / (sp.factorial(i) * sp.factorial((order - i - 1))))
            
        return None

    def coef_multon(self, order):
        """Coeficientes de Adams Multon de qualquer ordem"""
        
        x = sp.symbols('x')
        if order not in self.multon:
            self.multon[order] = []
            for i in reversed(range(order)):
                str_expr = '1'
                for j in range(order):
                    if j == order - i - 1:
                        continue
                    str_expr += f' * (x + {j} - 1)'
                expr = sp.sympify(str_expr)
                expr = sp.lambdify((x), expr)
                value = sp.integrate(expr(x), (x, 0, 1))
                self.multon[order].append(((-1)**(order - 1 - i) * value) / (sp.factorial(i) * sp.factorial((order - i - 1))))
            
    def coef_diff(self, order):
        """Coeficientes de Formula Inversa de qualquer ordem"""
        
        x = sp.symbols('x')
        if order not in self.diff:
            self.diff[order] = []
            for i in range(0, order + 1):
                str_expr = '1'
                for j in range(order + 1):
                    if j == i:
                        continue
                    str_expr += f' * (x - {j})/({i} - {j})' 
                expr = sp.sympify(str_expr)
                expr = sp.diff(expr, x)
                value = expr.evalf(subs={x: 0})
                if i == 0:
                    self.diff[order].append(-1/value)
                else:
                    self.diff[order].append(self.diff[order][0] * value)   

    def new_y_adam(self, h, res, index, order):
        """Obtém novo Y de Adams Bashforth"""
        
        self.coef_adam(order)
        y = res[index - 1][1]
        for i in range(1, order + 1):
            y += (h * self.adam[order][i - 1] 
                  * self.f(res[index - i][0], res[index - i][1])
                  )
        return y

    def new_y_multon(self, h, res, index, order):
        """Obtém novo Y de Adams Multon"""

        self.coef_multon(order)
        y = res[index - 1][1]
        y += (h * self.multon[order][0]
              * self.f(self.t0 + index*h, self.new_y_adam(h, res, index, order))
              )
        for i in range(2, order + 1):
            y += (h * self.multon[order][i - 1] 
                  * self.f(res[index - i + 1][0], res[index - i + 1][1])
                  )
        return y
    
    def new_y_inversa(self, h, res, index, order):
        """Obtém novo Y da Diferenciação Inversa"""

        self.coef_diff(order)
        k = self.new_y_adam(h, res, index, order)
        y = (h * self.diff[order][0] 
             * self.f(self.t0 + index*h , k)
             )
        for i in range(1, order + 1):
            y += self.diff[order][i] * res[index - i][1]
        
        return y

    def euler(self, h, nsteps):
        """Método de Euler simples"""
      
        res = []
        res.append([self.t0, self.y0])
        t, y = self.t0, self.y0
        for i in range(1, nsteps + 1):
            y = y + h*self.f(t, y)
            t = t + h
            res.append([t, y])
        
        _t = [x[0] for x in res]
        _y = [x[1] for x in res]
        plt.plot(_t, _y)
        return res

    def euler_inverso(self, h, nsteps):
        """Método de Euler Inverso"""

        res = []
        res.append([self.t0, self.y0])
        t, y = self.t0, self.y0
        for i in range(1, nsteps + 1):
            k = y + h*self.f(t, y)
            y = y + h*self.f(t + h, k)
            t = t + h
            res.append([t, y])

        _t = [x[0] for x in res]
        _y = [x[1] for x in res]
        plt.plot(_t, _y)
        return res

    def euler_aprimorado(self, h, nsteps):
        """Método de Euler Aprimorado"""

        res = []
        res.append([self.t0, self.y0])
        t, y = self.t0, self.y0
        for i in range(1, nsteps + 1):
            k = y + h*self.f(t, y)
            y = y + 0.5*h*(self.f(t, y) + self.f(t + h, k))
            t = t + h
            res.append((t,y))
        _t = [x[0] for x in res]
        _y = [x[1] for x in res]
        plt.plot(_t, _y)
        return res

        def runge_kutta(self, h, nsteps):
            """Método de Runge-Kutta"""

            res = []
            res.append([self.t0, self.y0])
            t, y = self.t0, self.y0
            for i in range(1, nsteps + 1):
                k1 = h * self.f(t, y)
                k2 = h * self.f(t + 0.5*h, y + 0.5*k1)
                k3 = h * self.f(t + 0.5*h, y + 0.5*k2)
                k4 = h * self.f(t + h, y + k3)
                y = y + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
                t = t + h
                res.append([t, y])
            _t = [x[0] for x in res]
            _y = [x[1] for x in res]
            plt.plot(_t, _y)
            return res


    def adam_bashfort(self, h, nsteps, order, preview):
        """Método de adam Bashfort utilizando previsão"""

        metodo = {'euler': self.euler, 'euler_inverso': self.euler_inverso,
                  'euler_aprimorado': self.euler_aprimorado, 
                  'runge_kutta': self.runge_kutta }
        res = metodo[preview](h, order - 1)
        t = self.t0 + (order - 1)*h
        for i in range(order, nsteps + 1):
            y = self.new_y_adam(h, res, i, order)
            t = t + h
            res.append([t, y])
        _t = [x[0] for x in res]
        _y = [x[1] for x in res]
        plt.plot(_t, _y)
        return res
    
    def adam_bashfort_lista(self, h, nsteps, order, lista):
        """Método de adam Bashfort utilizando lista"""

        res = lista
        t = self.t0 + (order - 1)*h
        for i in range(order, nsteps + 1):
            y = self.new_y_adam(h, res, i, order)
            t = t + h
            res.append([t, y])
        _t = [x[0] for x in res]
        _y = [x[1] for x in res]
        plt.plot(_t, _y)
        return res

    def adam_multon(self, h, nsteps, order, preview):
        """Método de Adams Multon utilizando previsão"""

        metodo = {'euler': self.euler, 'euler_inverso': self.euler_inverso,
                  'euler_aprimorado': self.euler_aprimorado, 
                  'runge_kutta': self.runge_kutta }
        res = metodo[preview](h, order - 1)
        t = self.t0 + (order - 1)*h
        for i in range(order, nsteps + 1):
            y = self.new_y_multon(h, res, i, order)
            t = t + h
            res.append([t, y])
        _t = [x[0] for x in res]
        _y = [x[1] for x in res]
        plt.plot(_t, _y)
        return res

    def adam_multon_lista(self, h, nsteps, order, lista):
        """Método de Adams Multon utilizando lista"""

        res = lista
        t = self.t0 + (order - 1)*h
        for i in range(order, nsteps + 1):
            y = self.new_y_multon(h, res, i, order)
            t = t + h
            res.append([t, y])
        _t = [x[0] for x in res]
        _y = [x[1] for x in res]
        plt.plot(_t, _y)
        return res
    
    def formula_inversa(self, h, nsteps, order, preview):
        """Método da Diferenciação Inversa utilizando previsão"""

        metodo = {'euler': self.euler, 'euler_inverso': self.euler_inverso,
                  'euler_aprimorado': self.euler_aprimorado, 
                  'runge_kutta': self.runge_kutta }
        res = metodo[preview](h, order - 1)
        t = self.t0 + (order - 1)*h
        for i in range(order, nsteps + 1):
            y = self.new_y_inversa(h, res, i, order)
            t = t + h
            res.append([t, y])
        _t = [x[0] for x in res]
        _y = [x[1] for x in res]
        plt.plot(_t, _y)
        return res

    def formula_inversa_lista(self, h, nsteps, order, lista):
        """Método da Diferenciação Inversa utilizando lista"""

        res = lista
        t = self.t0 + (order - 1)*h
        for i in range(order, nsteps + 1):
            y = self.new_y_inversa(h, res, i, order)
            t = t + h
            res.append([t, y])
        _t = [x[0] for x in res]
        _y = [x[1] for x in res]
        plt.plot(_t, _y)
        return res

    @staticmethod
    def _show():

        _str = 'Métodos.jpg'
        
        plt.legend(['Euler', 'Euler Inverso', 'Euler Aprimorado', 'Runge Kutta',
                    'Adam Bashforth', 'Adam Bashforth by Euler',
                    'Adam Bashforth by Euler Inverso', 
                    'Adam Bashforth by Euler Aprimorado',
                    'Adam Bashforth by Runge Kutta',
                    'Adam Multon',
                    'Adam Multon by Euler',
                    'Adam Multon by Euler Inverso',
                    'Adam Multon by Euler Aprimorado',
                    'Adam Multon by Runge Kutta',
                    'Formula Inversa',
                    'Formula Inversa by Euler',
                    'Formula Inversa by Euler Inverso',
                    'Formula Inversa by Euler Aprimorado',
                    'Formula Inversa by Runge Kutta'])
        plt.title('Gráfico dos Métodos Númericos')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(_str)

def main():
    """Código principal"""

    plt.figure(figsize=(12,8))
    plt.style.use('ggplot')
    with open('entrada.txt', 'r') as entrada, \
         open('saida.txt', 'w') as saida:
        for line in entrada:
            entrada = line.split()
            if not entrada:
                break
            metodo = entrada[0]
            if metodo in ['adam_bashforth', 'adam_multon', 'formula_inversa']:
                lista = []
                order = int(entrada[-1])
                str_expr = entrada[-2]
                nsteps = int(entrada[-3])
                h, t0 = float(entrada[-4]), float(entrada[-5])
                t = t0
                for i in range(order):
                    lista.append([t0, float(entrada[i + 1])])
                    t = t + h

            else:
                str_expr = entrada[5]
                t0, y0, h = float(entrada[1]), float(entrada[2]), float(entrada[3])
                nsteps = int(entrada[4])
            
            Solve = ODE(t0, y0, str_expr)
            saida.write(f'Método de {metodo}:\n')
            if metodo == 'euler':

                res = Solve.euler(h, nsteps)
            elif metodo == 'euler_inverso':

                res = Solve.euler_inverso(h, nsteps)
            elif metodo == 'euler_aprimorado':

                res = Solve.euler_aprimorado(h, nsteps)
            elif metodo == 'runge_kutta':

                res = Solve.runge_kutta(h, nsteps)

            elif metodo == 'adam_bashforth':
                
                res = Solve.adam_bashfort_lista(h, nsteps, order, lista)
            elif metodo == 'adam_bashforth_by_euler':

                order = int(entrada[6])
                res = Solve.adam_bashfort(h, nsteps, order, 'euler')

            elif metodo == 'adam_bashforth_by_euler_inverso':

                order = int(entrada[6])
                res = Solve.adam_bashfort(h, nsteps, order, 'euler_inverso')

            elif metodo == 'adam_bashforth_by_euler_aprimorado':

                order = int(entrada[6])
                res = Solve.adam_bashfort(h, nsteps, order, 'euler_aprimorado')

            elif metodo == 'adam_bashforth_by_runge_kutta':

                order = int(entrada[6])
                res = Solve.adam_bashfort(h, nsteps, order, 'runge_kutta')

            elif metodo == 'adam_multon':

                res = Solve.adam_multon_lista(h, nsteps, order, lista)
            elif metodo == 'adam_multon_by_euler':
                
                order = int(entrada[6])
                res = Solve.adam_multon(h, nsteps, order, 'euler')

            elif metodo == 'adam_multon_by_euler_inverso':
                
                order = int(entrada[6])
                res = Solve.adam_multon(h, nsteps, order, 'euler_inverso')

            elif metodo == 'adam_multon_by_euler_aprimorado':
               
                order = int(entrada[6])
                res = Solve.adam_multon(h, nsteps, order, 'euler_aprimorado')

            elif metodo == 'adam_multon_by_runge_kutta':
                
                order = int(entrada[6])
                res = Solve.adam_multon(h, nsteps, order, 'runge_kutta')

            elif metodo == 'formula_inversa':

                res = Solve.formula_inversa_lista(h, nsteps, order, lista)
            elif metodo == 'formula_inversa_by_euler':
              
                order = int(entrada[6])
                res = Solve.formula_inversa(h, nsteps, order, 'euler')

            elif metodo == 'formula_inversa_by_euler_inverso':
                
                order = int(entrada[6])
                res = Solve.formula_inversa(h, nsteps, order, 'euler_inverso')

            elif metodo == 'formula_inversa_by_euler_aprimorado':
                
                order = int(entrada[6])
                res = Solve.formula_inversa(h, nsteps, order, 'euler_aprimorado')

            elif metodo == 'formula_inversa_by_runge_kutta':
                
                order = int(entrada[6])
                res = Solve.formula_inversa(h, nsteps, order, 'runge_kutta')

            _t = [x[0] for x in res]
            _y = [x[1] for x in res]

            if metodo[0:18] == 'formula_inversa_by':
                metodo = 'formula_inversa_method'
            if metodo[0:14] == 'adam_multon_by':
                metodo = 'adam_multon_method'
            if metodo[0:17] == 'adam_bashforth_by':
                metodo = 'adam_bashforth_method'

            for i, r in enumerate(res):
                word = str(r[1])
                saida.write(f'{i:2}'+ ' ' + word + '\n')
            saida.write('\n')
        Solve._show()

if __name__ == '__main__':
    main()