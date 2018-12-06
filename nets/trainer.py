class Trainer():

  def __init__(model):

    self.model = model

    self.learningrate = 0.001
    self.momentum = 0.9
    self.weightdecay = 0.0004

    self.optimizer = 'sgd'
    self.loss = 'crossentropy'
    
  def 