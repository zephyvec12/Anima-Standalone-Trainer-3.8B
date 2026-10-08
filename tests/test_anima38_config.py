from pathlib import Path
import tempfile
import unittest

from anima38_config import build_native_config


class NativeConfigurationTests(unittest.TestCase):
    def test_preserves_user_training_and_prompts(self):
        config = {
            'model_arguments': {'dit_path': '/models/model.safetensors', 'qwen3_path': '/models/qwen3.safetensors',
                                'qwen35_path': '/models/custom_qwen35.safetensors', 'vae_path': '/models/vae.safetensors'},
            'training_arguments': {'learning_rate': 2e-5, 'optimizer_type': 'AdamW',
                                   'optimizer_args': ['weight_decay=0.01', 'betas=(0.8,0.95)'],
                                   'lr_scheduler': 'constant', 'lr_warmup_steps': 100,
                                   'max_train_steps': 11982, 'gradient_accumulation_steps': 1,
                                   'save_last_n_steps_state': 749, 'blocks_to_swap': 0},
            'network_arguments': {'network_dim': 32, 'network_alpha': 32},
            'sample_arguments': {'sample_prompts': '/job/user_prompts.txt', 'sample_every_n_steps': 250},
        }
        allowed = {key for group in config.values() for key in group}
        native = build_native_config(config, allowed, '/cache/connector.safetensors')
        self.assertEqual(native['max_train_steps'], 11982)
        self.assertEqual(native['blocks_to_swap'], 0)
        self.assertEqual(native['network_dim'], 32)
        self.assertEqual(native['network_alpha'], 32)
        self.assertEqual(native['gradient_accumulation_steps'], 1)
        self.assertEqual(native['save_last_n_steps_state'], 749)
        self.assertEqual(native['optimizer_type'], 'AdamW')
        self.assertIn('betas=(0.8,0.95)', native['optimizer_args'])
        self.assertNotIn('betas=(0.9,0.99)', native['optimizer_args'])
        self.assertEqual(native['sample_prompts'], '/job/user_prompts.txt')
        self.assertEqual(native['qwen35'], '/models/custom_qwen35.safetensors')
        self.assertEqual(native['lr_scheduler'], 'constant_with_warmup')
        self.assertEqual(native['text_encoder_lr'], 0)


if __name__ == '__main__':
    unittest.main()
