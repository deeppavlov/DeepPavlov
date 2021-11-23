from joblib import logger
import tensorflow as tf
import datetime
import time
from itertools import islice
from typing import List, Tuple, Union, Optional, Iterable
import json
from deeppavlov.core.trainers.utils import parse_metrics, NumpyArrayEncoder


class TrainLogger:
    def __init__(self,log_dir):
        pass
    
    def __call__(self,metrics):
        pass

class TensorboardLogger(TrainLogger):
    def __init__(self, type, log_dir):
        self.tb_writer = tf.summary.FileWriter(log_dir)
        self.type = type
        #self.tb_train_writer = tf.summary.FileWriter(str(log_dir / 'train_log'))
        #self.tb_valid_writer = tf.summary.FileWriter(str(log_dir / 'valid_log'))

    def __call__(self, nn_trainer, iterator, tensorboard_tag: Optional[str] = None, tensorboard_index: Optional[int] = None , log = None ): # default value for log for now = None
        if self.type =='train':
            print("logging Training metrics...")
            nn_trainer._send_event(event_name='before_log')
            if nn_trainer.log_on_k_batches == 0:
                report = {
                    'time_spent': str(datetime.timedelta(seconds=round(time.time() - nn_trainer.start_time + 0.5)))
                }
            else:
                data = islice(iterator.gen_batches(nn_trainer.batch_size, data_type='train', shuffle=True),
                            nn_trainer.log_on_k_batches)
                report = nn_trainer.test(data, nn_trainer.train_metrics, start_time=nn_trainer.start_time)

            report.update({
                'epochs_done': nn_trainer.epoch,
                'batches_seen': nn_trainer.train_batches_seen,
                'train_examples_seen': nn_trainer.examples
            })

            metrics: List[Tuple[str, float]] = list(report.get('metrics', {}).items()) + list(nn_trainer.last_result.items())

            report.update(nn_trainer.last_result)
            if nn_trainer.losses:
                report['loss'] = sum(nn_trainer.losses) / len(nn_trainer.losses)
                nn_trainer.losses.clear()
                metrics.append(('loss', report['loss']))

            # if metrics and self.tensorboard_log_dir is not None:
            # if metrics and nn_trainer.tensorboard_idx is not None:
            #     self.TensorboardLogger_train(self, metrics, tensorboard_tag, tensorboard_index)
            if metrics and nn_trainer.tensorboard_idx is not None:
                summary = nn_trainer._tf.Summary()

                for name, score in metrics:
                    summary.value.add(tag=f'{tensorboard_tag}/{name}', simple_value=score)
                self.tb_writer.add_summary(summary, tensorboard_index)
                self.tb_writer.flush()
                # summary = self._tf.Summary()

                # for name, score in metrics:
                #     summary.value.add(tag=f'{tensorboard_tag}/{name}', simple_value=score)
                
                # self.tb_writer.add_summary(summary, tensorboard_index)
                # self.tb_writer.flush()
                # self.TensorboardLogger_train(summary,tensorboard_index)
                # self.TensorboardLogger('train',summary,tensorboard_index)
                #self.tb_train_writer.add_summary(summary, tensorboard_index)
                #self.tb_train_writer.flush()

            nn_trainer._send_event(event_name='after_train_log', data=report)
            report = {'train': report}
            print(json.dumps(report, ensure_ascii=False, cls=NumpyArrayEncoder))
        else:
            print("logging Validation metrics...")
            nn_trainer._send_event(event_name='before_validation')
            report = nn_trainer.test(iterator.gen_batches(nn_trainer.batch_size, data_type='valid', shuffle=False),
                            start_time=nn_trainer.start_time)

            report['epochs_done'] = nn_trainer.epoch
            report['batches_seen'] = nn_trainer.train_batches_seen
            report['train_examples_seen'] = nn_trainer.examples

            metrics = list(report['metrics'].items())

            if tensorboard_tag is not None and nn_trainer.tensorboard_idx is not None:
                summary = nn_trainer._tf.Summary()
                for name, score in metrics:
                    summary.value.add(tag=f'{tensorboard_tag}/{name}', simple_value=score)
                if tensorboard_index is None:
                    tensorboard_index = nn_trainer.train_batches_seen
                self.tb_writer.add_summary(summary, tensorboard_index)
                self.tb_writer.flush()

            m_name, score = metrics[0]

            # Update the patience
            if nn_trainer.score_best is None:
                nn_trainer.patience = 0
            else:
                if nn_trainer.improved(score, nn_trainer.score_best):
                    nn_trainer.patience = 0
                else:
                    nn_trainer.patience += 1

            # Run the validation model-saving logic
            if nn_trainer._is_initial_validation():
                log.info('Initial best {} of {}'.format(m_name, score))
                nn_trainer.score_best = score
            elif nn_trainer._is_first_validation() and nn_trainer.score_best is None:
                log.info('First best {} of {}'.format(m_name, score))
                nn_trainer.score_best = score
                log.info('Saving model')
                nn_trainer.save()
            elif nn_trainer.improved(score, nn_trainer.score_best):
                log.info('Improved best {} of {}'.format(m_name, score))
                nn_trainer.score_best = score
                log.info('Saving model')
                nn_trainer.save()
            else:
                log.info('Did not improve on the {} of {}'.format(m_name, nn_trainer.score_best))

            report['impatience'] = nn_trainer.patience
            if nn_trainer.validation_patience > 0:
                report['patience_limit'] = nn_trainer.validation_patience

            nn_trainer._send_event(event_name='after_validation', data=report)
            report = {'valid': report}
            print(json.dumps(report, ensure_ascii=False, cls=NumpyArrayEncoder))
            nn_trainer.validation_number += 1



        # summary = tf.Summary()
        # for name, score in metrics:
        #     summary.value.add(tag=f'{tensorboard_tag}/{name}', simple_value=score)
        # self.tb_writer.add_summary(summary, tensorboard_index)
        # self.tb_writer.flush()


        # if train_or_valid == 'train':
        #     self.tb_train_writer.add_summary(summary, tensorboard_index)
        #     self.tb_train_writer.flush()
        # else:
        #     self.tb_valid_writer.add_summary(summary, tensorboard_index)
        #     self.tb_valid_writer.flush()