#!/usr/bin/env python
'''
'''
import unittest
import logging
from logging import config

config.fileConfig("conf/nnts.cfg")
logger = logging.getLogger()

class ModuleImportTestCase(unittest.TestCase):
	def test_TensorFlow(self):
		test_result = False
		logger.info("try to test TensorFlow installation...")
		try:
			import tensorflow
			logger.info("succeed in importing TensorFlow")
			test_result = True
		except:
			logger.error("Failed to import TensorFlow")
			raise
		self.assertEqual(True, test_result)

  	def test_keras(self):
		test_result = False
		logger.info("try to test keras installation...")
		try:
			import keras
			logger.info("succeed in importing keras")
			test_result = True
		except:
			logger.error("Failed to import keras")
			raise
		self.assertEqual(True, test_result)

if __name__=='__main__':
	unittest.main()
