# deep_learning_briscola

Briscola is a popular italian card game. 

The main goal of this repository is to train a neural network to play briscola.

$E[Y|(s,a)] =  \sum_{s'} (R + \gamma \operatorname{max}_{a'} E[Y|(s',a')])p(s'| s,a)$. We approximate the sum with $ R_{s'} + \gamma \operatorname{max}_{a'} E[Y|(s',a')] $

