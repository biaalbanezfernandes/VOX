# VOX — Segurança digital por voz

Projeto escolar de reconhecimento de comandos falados com MFCCs e uma rede neural artificial feita em TensorFlow/Keras. A ideia é reconhecer quatro comandos e mostrar uma ação **simulada**, sem controlar sistemas reais.

## Comandos do VOX

- **validar** — simula o início de uma verificação.
- **bloquear** — simula a ativação de um bloqueio na interface.
- **cancelar** — simula o cancelamento de uma operação.
- **status** — mostra o status fictício do sistema.

> Este projeto é uma demonstração educacional, não um mecanismo real de autenticação ou segurança. Reconhecer uma palavra não comprova a identidade de quem falou.

## 1. Requisitos

Recomendado: Python 3.10 ou 3.11. Microfone conectado para gravar e testar.

Abra o terminal dentro desta pasta e crie um ambiente virtual:

### Windows
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se o TensorFlow não instalar na versão de Python instalada, use uma versão compatível (por exemplo, Python 3.10/3.11) e crie o ambiente virtual novamente.

## 2. Gravar o conjunto de dados

Execute:

```bash
python gravar_amostras.py
```

O programa cria pastas em `dataset/`, uma para cada comando. Grave pelo menos **20 amostras por comando**; se possível, cada integrante deve gravar várias amostras. Fale uma palavra por gravação, em volume normal, variando um pouco a entonação e a distância do microfone.

Estrutura esperada:
```text
dataset/
  validar/
    validar_001.wav
  bloquear/
    bloquear_001.wav
  cancelar/
    cancelar_001.wav
  status/
    status_001.wav
```

O script permite parar uma classe digitando `p`. Caso uma classe fique com menos de cinco amostras, o treinamento será interrompido e avisará.

## 3. Treinar e avaliar

```bash
python treinar_modelo.py
```

O programa:
1. carrega os áudios com Librosa;
2. extrai 13 MFCCs e resume cada coeficiente pela média e pelo desvio padrão (26 características);
3. separa 75% das amostras para treino/validação e 25% para teste; a validação ocorre dentro do conjunto de treino;
4. padroniza as características sem usar o conjunto de teste para ajustar a padronização;
5. treina uma rede neural multiclasses com saída softmax;
6. calcula a acurácia e o relatório por classe;
7. salva o modelo e os gráficos em `modelos/` e `resultados/`.

Arquivos gerados:
- `modelos/vox_model.keras`
- `modelos/classes.json`
- `modelos/scaler.npz`
- `resultados/acuracia.png`
- `resultados/matriz_confusao.png`
- `resultados/avaliacao.txt`

## 4. Testar com o microfone

Depois do treinamento:

```bash
python vox.py
```

Pressione ENTER, diga um dos comandos e confira a previsão. A aplicação mostra a classe prevista, a confiança estimada e a ação simulada. A confiança não é uma garantia de que a previsão esteja correta. O limiar inicial de 60% é heurístico e pode precisar de ajuste.

## 5. Boas práticas para uma avaliação honesta

- Não use apenas áudios gerados por TTS repetidos; grave vozes reais.
- Não use as mesmas gravações no treino e no teste.
- Tenha quantidade semelhante de exemplos por classe.
- Grave em condições diferentes e teste com alguém que não gravou todos os áudios.
- Registre a acurácia real e mostre a matriz de confusão.
- Se duas palavras forem confundidas, discuta ruído, pronúncia, quantidade de amostras e semelhanças acústicas.
- Um conjunto pequeno pode produzir métricas instáveis. Acurácia alta não significa que o sistema esteja pronto para uso real.

## 6. Estrutura do projeto

```text
VOX_Seguranca_por_Voz/
├── dataset/                 # áudios gravados pelo grupo
├── modelos/                 # arquivos criados no treinamento
├── resultados/              # gráficos e métricas
├── vox_config.py
├── gravar_amostras.py
├── treinar_modelo.py
├── vox.py
├── requirements.txt
├── RELATORIO.md
└── ROTEIRO_APRESENTACAO.md
```

## Solução de problemas

- **Microfone não grava:** verifique a permissão de microfone do sistema e o dispositivo padrão.
- **`pip` não instala TensorFlow:** verifique a versão do Python; tente Python 3.10/3.11.
- **Acurácia baixa:** colete mais exemplos, balanceie as classes, reduza ruído e teste pronúncias variadas.
- **O programa prevê comandos errados:** isso é esperado em um classificador simples; não esconda os erros na apresentação.
