# Métodos Numéricos(IF816) EC-2018.2
Autor: Mateus Caracciolo Marques\
Login: mcm2\
\
Projeto da cadeira de Métodos Numéricos(IF816) escrito na linguagem Python, versão 3.7.0, contendo diversos métodos para se resolver a equação diferencial de 1 ordem y'(t) = f(t, y(t)) com valor inicial y(t0) = y0.\
\
Bibliotecas usadas:\
\
1- SymPy: https://www.sympy.org/en/index.html
\
2- Matplotlib: https://matplotlib.org
\
Como executar o código:

1. Antes de tudo a entrada deve se encontrar em um arquivo com o nome "entrada.txt" com a formatação fornecida pelo projeto(veja o exemplo abaixo para ver essa formatação).

2. Execute no terminal o comando "python3 main.py" para executar o comando.

3. A saída será criada e disponível em um arquivo chamado "saida.txt". Também será criado um arquivo "Métodos.jpg" correspondendo aos gráficos de cada método.

Exemplos de código:

.Arquivo de entrada: "entrada.txt":

euler 0 0 0.1 20 1-t+4*y  \
euler_inverso 0 0 0.1 20 1-t+4*y  \
euler_aprimorado 0 0 0.1 20 1-t+4*y  \
runge_kutta 0 0 0.1 20 1-t+4*y  \
adam_bashforth 0.0 0.1 0.23 0.402 0.6328 0 0.1 20 1-t+4*y 5  \
adam_bashforth_by_euler 0 0 0.1 20 1-t+4*y 6  \
adam_bashforth_by_euler_inverso 0 0 0.1 20 1-t+4*y 6  \
adam_bashforth_by_euler_aprimorado 0 0 0.1 20 1-t+4*y 6  \
adam_bashforth_by_runge_kutta 0 0 0.1 20 1-t+4*y 6  \
adam_multon 0.0 0.1 0.23 0.402 0.6328 0 0.1 20 1-t+4*y 5  \
adam_multon_by_euler 0 0 0.1 20 1-t+4*y 6  \
adam_multon_by_euler_inverso 0 0 0.1 20 1-t+4*y 6  \
adam_multon_by_euler_aprimorado 0 0 0.1 20 1-t+4*y 6  \
adam_multon_by_runge_kutta 0 0 0.1 20 1-t+4*y 6  \
formula_inversa 0.0 0.1 0.23 0.402 0.6328 0 0.1 20 1-t+4*y 5  \
formula_inversa_by_euler 0 0 0.1 20 1-t+4*y 6  \
formula_inversa_by_euler_inverso 0 0 0.1 20 1-t+4*y 6  \
formula_inversa_by_euler_aprimorado 0 0 0.1 20 1-t+4*y 6  \
formula_inversa_by_runge_kutta 0 0 0.1 20 1-t+4*y 7  
  
.Arquivo de saída: "saida.txt":  
  
Método de euler:  \
 0 0.0  \
 1 0.1  \
 2 0.23  \
 3 0.402  \
 4 0.6328  \
 5 0.9459200000000001  \
 6 1.3742880000000002  \
 7 1.9640032000000003  \
 8 2.7796044800000006  \
 9 3.911446272000001  \
10 5.486024780800001  \
11 7.680434693120002  \
12 10.742608570368002  \
13 15.019651998515204  \
14 20.99751279792129  \
15 29.356517917089803  \
16 41.04912508392572  \
17 57.40877511749601  \
18 80.30228516449442  \
19 112.34319923029219  \
20 157.19047892240906  

Método de euler_inverso:  \
 0 0.0  \
 1 0.13  \
 2 0.31880000000000003  \
 3 0.5993280000000001  \
 4 1.0229516800000003  \
 5 1.6698046208000004  \
 6 2.6648952084480007  \
 7 4.203236525178881  \
 8 6.589048979279054  \
 9 10.296916407675326  \
10 16.06718959597351  \
11 25.054815769718676  \
12 39.061512600761134  \
13 60.89795965718737  \
14 94.9488170652123  \
15 148.0541546217312  \
16 230.88448120990066  \
17 360.085790687445  \
18 561.6258334724142  \
19 876.0143002169661  \
20 1366.4463083384671  

Método de euler_aprimorado:    \
 0 0.0  \
 1 0.11499999999999999  \
 2 0.2732  \
 3 0.495336  \
 4 0.8120972800000001  \
 5 1.2689039744000001  \
 6 1.9329778821120003  \
 7 2.9038072655257605  \
 8 4.328634752978125  \
 9 6.425379434407626  \
10 9.516561562923286  \
11 14.079511113126465  \
12 20.820676447427168    \
13 30.78560114219221\
14 45.521689690444475\
15 67.31910074185782\
16 99.56726909794958\
17 147.28255826496536\
18 217.88918623214875\
19 322.37499562358016\
20 477.00199352289866

Método de runge_kutta:\
 0 0.0\
 1 0.11719999999999998\
 2 0.2797378133333333\
 3 0.5099075540764444\
 4 0.8409660953343013\
 5 1.3225238232800218\
 6 2.028586204647585\
 7 3.0695496610129576\
 8 4.610096214321729\
 9 6.895887526110867\
10 10.293385285617116\
11 15.349252610064573\
12 22.878965093520325\
13 34.09899486217405\
14 50.82399393573377\
15 75.76093922039858\
16 112.94791839970925\
17 168.4086814741263\
18 251.12905711100333\
19 374.513505461054\
20 558.557906546436211

Método de adam_bashforth:\
 0 0.0\
 1 0.1\
 2 0.23\
 3 0.402\
 4 0.6328\
 5 1.0450693333333334\
 6 1.5769452237037038\
 7 2.433243745700412\
 8 3.664461237935084\
 9 5.453975591981679\
10 8.137554454618165\
11 12.14491781219452\
12 18.087547487492614\
13 26.925704755555962\
14 40.10388059172935\
15 59.74360121151783\
16 89.00354768594437\
17 132.61375467426444\
18 197.62662975430834\
19 294.54367115875914\
20 439.0230902223277

Método de adam_bashforth_by_euler:\
 0 0.0\
 1 0.1\
 2 0.23\
 3 0.402\
 4 0.6328\
 5 0.9459200000000001\
 6 1.4503504\
 7 2.2369193168888897\
 8 3.358441744426052\
 9 4.996950834981482\
10 7.481986396679563\
11 11.180288833358992\
12 16.624920205036048\
13 24.750058130421603\
14 36.913899790025475\
15 55.006224132263085\
16 81.93204430281415\
17 122.13841135433535\
18 182.12506888649392\
19 271.5127191821949\
20 404.8197598488207

Método de adam_bashforth_by_euler_inverso:\
 0 0.0\
 1 0.13\
 2 0.31880000000000003\
 3 0.5993280000000001\
 4 1.0229516800000003\
 5 1.6698046208000004\
 6 2.5666957037226674\
 7 3.8351793259402913\
 8 5.760861808339608\
 9 8.649737844496507\
10 12.882727090398287\
11 19.176216262218144\
12 28.622431363397315\
13 42.68267074275497\
14 63.57155794218719\
15 94.75900710179582\
16 141.3232887354071\
17 210.69586797430568\
18 314.1162509000865\
19 468.44535675120824\
20 698.6313848849245

Método de adam_bashforth_by_euler_aprimorado:\
 0 0.0\
 1 0.11499999999999999\
 2 0.2732\
 3 0.495336\
 4 0.8120972800000001\
 5 1.2689039744000001\
 6 1.9456653358079998\
 7 2.950345550541227\
 8 4.430330733618685\
 9 6.62178245644458\
10 9.886678943910466\
11 14.744965826995497\
12 21.96943623536663\
13 32.73516478160922\
14 48.789660186960745\
15 72.71728424412105\
16 108.38614825552158\
17 161.58444762054373\
18 240.9243942843869\
19 359.23897662227506\
20 535.69606313874

Método de adam_bashforth_by_runge_kutta:\
 0 0.0\
 1 0.11719999999999998\
 2 0.2797378133333333\
 3 0.5099075540764444\
 4 0.8409660953343013\
 5 1.3225238232800218\
 6 2.0283613404798513\
 7 3.0687222083470225\
 8 4.608280207362133\
 9 6.892365150866892\
10 10.286722882121998\
11 15.337228379453155\
12 22.858123524780787\
13 34.06348461117281\
14 50.764240868303375\
15 75.66185037267206\
16 112.78540154709049\
17 168.14409639982503\
18 250.70125222195327\
19 373.8262077743362\
20 557.4593489350084

Método de adam_multon:\
 0 0.0\
 1 0.1\
 2 0.23\
 3 0.402\
 4 0.6328\
 5 1.0450693333333334\
 6 1.5769452237037038\
 7 2.433243745700412\
 8 3.664461237935084\
 9 5.453975591981679\
10 8.137554454618165\
11 12.14491781219452\
12 18.087547487492614\
13 26.925704755555962\
14 40.10388059172935\
15 59.74360121151783\
16 89.00354768594437\
17 132.61375467426444\
18 197.62662975430834\
19 294.54367115875914\
20 439.0230902223277

Método de adam_multon_by_euler:\
 0 0.0\
 1 0.1\
 2 0.23\
 3 0.402\
 4 0.6328\
 5 0.9459200000000001\
 6 1.4604877444444448\
 7 2.22993722678856\
 8 3.352539966518789\
 9 5.020331512540322\
10 7.498066876435405\
11 11.178845091167284\
12 16.658425805879407\
13 24.821313929847058\
14 36.98580590532824\
15 55.12068465992721\
16 82.16231807629735\
17 122.49078722202131\
18 182.64078710167408\
19 272.36073113752565\
20 406.1932529422069

Método de adam_multon_by_euler_inverso:\
 0 0.0\
 1 0.13\
 2 0.31880000000000003\
 3 0.5993280000000001\
 4 1.0229516800000003\
 5 1.6698046208000004\
 6 2.5545018176082963\
 7 3.844999963243992\
 8 5.773261958767941\
 9 8.631410527689026\
10 12.880401149276723\
11 19.21058356294343\
12 28.640734065172072\
13 42.69554491112505\
14 63.65109581046852\
15 94.9003337162408\
16 141.50564341850608\
17 211.01963243976428\
18 314.7087946203177\
19 469.38066297531697\
20 700.1089756566623

Método de adam_multon_by_euler_aprimorado:\
 0 0.0\
 1 0.11499999999999999\
 2 0.2732\
 3 0.495336\
 4 0.8120972800000001\
 5 1.2689039744000001\
 6 1.9476419545493335\
 7 2.9502203842872285\
 8 4.4314564475511515\
 9 6.629743388804297\
10 9.897206620645782\
11 14.758809530229122\
12 21.99922483138544\
13 32.7883490981638\
14 48.87123687402625\
15 72.85152188197434\
16 108.6132153674694\
17 161.95044378691287\
18 241.50700039700826\
19 360.17774248202477\
20 537.1992891570892

Método de adam_multon_by_runge_kutta:\
 0 0.0\
 1 0.11719999999999998\
 2 0.2797378133333333\
 3 0.5099075540764444\
 4 0.8409660953343013\
 5 1.3225238232800218\
 6 2.028681431010557\
 7 3.069852178407987\
 8 4.61076912669213\
 9 6.897230232241839\
10 10.295895976303225\
11 15.353750812854953\
12 22.88679986208732\
13 34.1123608854548\
14 50.84643573085178\
15 75.79815095391656\
16 113.00899987035763\
17 168.50811012807696\
18 251.28977714919586\
19 374.77175387892225\
20 558.9707382058023

Método de formula_inversa:\
 0 0.0\
 1 0.1\
 2 0.23\
 3 0.402\
 4 0.6328\
 5 0.9913990072992702\
 6 1.5294009917516476\
 7 2.3387400169735435\
 8 3.5242015797349504\
 9 5.270657841458439\
10 7.866067380257344\
11 11.731575020741472\
12 17.48723520641364\
13 26.059557480261837\
14 38.83533205040848\
15 57.88439145298088\
16 86.29289829193638\
17 128.66416045067038\
18 191.86656848830546\
19 286.1480016154262\
20 426.7975881360264

Método de formula_inversa_by_euler:\
 0 0.0\
 1 0.1\
 2 0.23\
 3 0.402\
 4 0.6328\
 5 0.9459200000000001\
 6 1.4233306775510208\
 7 2.17028715532084\
 8 3.286852229335917\
 9 4.91951229910342\
10 7.331615996823956\
11 10.931486355502962\
12 16.300754949307862\
13 24.29067071438677\
14 36.18810578265492\
15 53.9286358440705\
16 80.3907040246658\
17 119.85489337660272\
18 178.71097993745062\
19 266.5024007030086\
20 397.4660671869962

Método de formula_inversa_by_euler_inverso:\
 0 0.0\
 1 0.13\
 2 0.31880000000000003\
 3 0.5993280000000001\
 4 1.0229516800000003\
 5 1.6698046208000004\
 6 2.599750369913906\
 7 3.9188002743911277\
 8 5.855154553136913\
 9 8.756306250754074\
10 13.086945844243083\
11 19.51934220459219\
12 29.088978934175962\
13 43.36210910590688\
14 64.65573199502246\
15 96.40643047551318\
16 143.75126810531052\
17 214.37096855885352\
18 319.72016737693951\
19 476.8740862842272\
20 711.3063220118161

Método de formula_inversa_by_euler_aprimorado:\
 0 0.0\
 1 0.11499999999999999\
 2 0.2732\
 3 0.495336\
 4 0.8120972800000001\
 5 1.2689039744000001\
 6 1.9415103594788574\
 7 2.9403792822341073\
 8 4.420767449562214\
 9 6.61356800032001\
10 9.87064022815231\
11 14.719631602018772\
12 21.943196586036024\
13 32.70609493558996\
14 48.748720758987666\
15 72.6701628644441\
16 108.34641229238731\
17 161.55756363826256\
18 240.92722022482042\
19 359.32231200324577\
20 535.9383197079903

Método de formula_inversa_by_runge_kutta:\
 0 0.0\
 1 0.11719999999999998\
 2 0.2797378133333333\
 3 0.5099075540764444\
 4 0.8409660953343013\
 5 1.3225238232800218\
 6 2.028586204647585\
 7 3.07922210154974\
 8 4.65388795651104\
 9 7.00724576838788\
10 10.5183792193527\
11 15.7713607926875\
12 23.6481049693525\
13 35.4570842314832\
14 53.1548537264175\
15 79.6921082138951\
16 119.504301699474\
17 179.231131497329\
18 268.824895152577\
19 403.233370399925\
20 604.896002627814
