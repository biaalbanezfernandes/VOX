# Roteiro de apresentação — VOX
**Data:** 28/10/2026 | **Duração sugerida:** 5–7 minutos

## 1. Introdução (30–45 s)
“Nosso projeto se chama VOX e utiliza uma rede neural para reconhecer comandos de voz relacionados a uma interface de segurança digital. Ele é uma demonstração educacional: as ações são simuladas e não alteram sistemas reais.”

## 2. Comandos e objetivo (30 s)
“Treinamos quatro classes: validar, bloquear, cancelar e status. Cada gravação é identificada com a classe correta, que funciona como rótulo para o aprendizado supervisionado.”

## 3. Como a IA aprende (1 min)
“Primeiro gravamos exemplos de voz. O Librosa carrega os arquivos e extrai MFCCs, que resumem características do som. A rede neural do TensorFlow/Keras aprende relações entre esses valores e os comandos. Depois avaliamos o modelo com áudios que ficaram separados do treinamento.”

## 4. Demonstração ao vivo (1–2 min)
1. Abrir o terminal na pasta do projeto.
2. Executar `python vox.py`.
3. Pressionar ENTER e falar “validar”.
4. Repetir com “cancelar” e “status”.
5. Se houver erro, mostrar honestamente e explicar que o desempenho depende dos dados e do ambiente.

## 5. Resultados (1 min)
Mostrar a acurácia medida no teste, `resultados/acuracia.png` e `resultados/matriz_confusao.png`. Explicar quais classes foram reconhecidas melhor e quais foram confundidas. Não afirmar 100% se esse não for o resultado obtido.

## 6. Limitações e conclusão (45 s)
“O modelo pode errar com ruído ou pronúncias diferentes e o conjunto de dados é limitado. Reconhecer uma palavra não prova quem está falando. Para uso real em segurança, seria necessário combinar outras formas de autenticação e fazer testes muito mais rigorosos.”

## Perguntas que podem aparecer
- **O que são MFCCs?** Características numéricas que representam aspectos do espectro do som em escala Mel.
- **Por que separar treino e teste?** Para medir se o modelo funciona em exemplos que não foram usados para ajustar seus pesos.
- **O que significa acurácia?** A proporção de previsões corretas entre as amostras avaliadas.
- **Por que o modelo erra?** Pode haver ruído, pouca diversidade de gravações, diferenças de pronúncia ou semelhança acústica.
- **O VOX autentica a pessoa?** Não. Ele classifica palavras, não a identidade de quem falou.
