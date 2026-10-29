> **Registro histórico — não é o plano vigente.** Arquivado em 28/10/2026.
> O [estado atual](../PROJECT_STATUS.md), os [resultados corrigidos da 0015](../results/CAMPAIGN_0015.md)
> e a [validação vigente](../../VALIDATION.md) prevalecem sobre descrições e conclusões antigas.

# Nomes numéricos de campanhas — 27/10/2026

Pedido: substituir o identificador aleatório dos ZIPs por números legíveis.

Plano: helper stdlib em src/llm2/research/naming.py, usado pelo bootstrap fixado
e defaults CLI de treino/revisão. Gerar llm2-campaign-0001 e ZIP correspondente;
revisões usam llm2-review-0001. Detectar números em pastas/ZIPs existentes e,
no Kaggle, nos outputs anteriores anexados em /kaggle/input. Reservar cada
número por mkdir exclusivo em registro separado, sem pré-criar a pasta da campanha.
Aceitar run_number inteiro positivo opcional para continuidade manual entre sessões.

Riscos: sessão nova sem arquivos anteriores não tem histórico e reinicia em 0001;
não há contador global remoto. Erros podem consumir um número, deixando lacunas.
Nunca reutilizar ou sobrescrever dados/checkpoints/ZIPs; pedido explícito de um
número ocupado deve falhar. O checkout Git mantém seu SHA para proveniência.

Aceitação: nomes previsíveis, avanço após ZIP/pasta/reserva existente, consulta
a outputs anexados, números manuais validados, colisões concorrentes sem reuso,
registro não interfere no mkdir exclusivo da campanha. Sem alteração nos modelos,
treino, pesos ou dados. Testar helper e integrações existentes relevantes; documentar
resultados reais e publicar novo commit para atualizar a célula do Kaggle.
