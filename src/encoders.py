# -*- coding: utf-8 -*-
from state_transfer_rnn import *

class LSTMEncoder(StateTransferLSTM):

	def __init__(self, decoder=None, decoders=[], **kwargs):
		super(LSTMEncoder, self).__init__(**kwargs)
		if decoder:
			decoders = [decoder]
		self.broadcast_state(decoders)