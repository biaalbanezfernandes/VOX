# VOX — Reconhecimento de comandos de voz para segurança digital

**Integrantes:** [preencher]  
**Turma:** [preencher]  
**Disciplina:** [preencher]  
**Data da apresentação:** 28/10/2026

## 1. Objetivo

Desenvolver um programa em Python que utilize uma rede neural artificial para classificar comandos falados relacionados a uma interface demonstrativa de segurança digital. O sistema reconhece as classes `validar`, `bloquear`, `cancelar` e `status`, e responde com ações simuladas.

## 2. Fundamentação teórica

### Rede neural artificial
Uma rede neural artificial é um modelo computacional composto por camadas de unidades conectadas que transformam entradas numéricas em previsões. Durante o treinamento, seus pesos são ajustados para reduzir a diferença entre as previsões e as classes corretas.

### Aprendizado supervisionado
No aprendizado supervisionado, cada exemplo possui uma entrada e um rótulo conhecido. Neste projeto, cada áudio gravado é associado à palavra que foi pronunciada. O modelo utiliza exemplos rotulados para aprender padrões e é avaliado com áudios separados do treinamento.

### MFCCs
Os coeficientes cepstrais de frequência de Mel (MFCCs) são características usadas no processamento de áudio. Eles representam aspectos do espectro sonoro numa escala de frequências inspirada na percepção auditiva humana. O projeto calcula 13 coeficientes por quadro e resume cada coeficiente com média e desvio padrão, formando um vetor de 26 valores por gravação.

## 3. Coleta e tratamento dos dados

Os áudios foram gravados pelos integrantes usando `sounddevice` e salvos em formato WAV. O Librosa carrega os arquivos, converte-os para mono e para uma taxa de amostragem de 22.050 Hz, ajusta a duração e extrai os MFCCs.

**Preencher após a coleta:**

| Comando | Quantidade de gravações |
|---|---:|
| validar | [preencher] |
| bloquear | [preencher] |
| cancelar | [preencher] |
| status | [preencher] |
| **Total** | [preencher] |

## 4. Modelo e treinamento

A rede foi construída com TensorFlow/Keras. A entrada possui 26 características; há duas camadas densas ocultas com ativações ReLU e uma camada de saída softmax com quatro classes. Foi usado o otimizador Adam e a função de perda `sparse_categorical_crossentropy`.

Os dados foram divididos em treino e teste, mantendo a proporção entre classes. O padronizador foi ajustado apenas com os dados de treino. Uma parte do treino foi utilizada para validação durante o ajuste do modelo.

## 5. Resultados

**Preencher com os valores realmente produzidos pelo programa. Não inventar resultados.**

- Acurácia no teste: [preencher]%
- Classe mais reconhecida: [preencher]
- Confusão mais frequente: [preencher]
- Quantidade de amostras de teste: [preencher]

Inserir os gráficos gerados em `resultados/acuracia.png` e `resultados/matriz_confusao.png`.

## 6. Discussão de erros e limitações

O modelo pode confundir palavras devido a ruído, diferenças de pronúncia, volume, distância do microfone ou quantidade limitada de exemplos. A média e o desvio padrão dos MFCCs resumem o áudio e podem descartar detalhes temporais. A saída do modelo é uma estimativa, não uma certeza.

O projeto classifica comandos falados, mas **não verifica a identidade da pessoa**. Portanto, não deve ser usado para liberar contas, validar pagamentos ou substituir controles reais de segurança.

## 7. Aplicações reais e futuras

O reconhecimento de comandos pode apoiar interfaces sem uso das mãos, acessibilidade e navegação por voz. Como evolução, seria possível aumentar o conjunto de dados, avaliar diferentes falantes, comparar modelos, usar características temporais mais ricas e implementar uma interface visual. Aplicações de segurança real exigiriam mecanismos adicionais de autenticação, proteção contra gravações reproduzidas e testes rigorosos.

## 8. Conclusão

O projeto demonstra o fluxo de uma tarefa de classificação de áudio: coleta de dados rotulados, extração de MFCCs, treinamento supervisionado e avaliação em amostras separadas. O desempenho final deve ser interpretado com base nas métricas reais e nas limitações do conjunto de dados.
