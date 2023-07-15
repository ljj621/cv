class AverageValueMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, count=1):
        # if count == 0 and self.count == 0:
        #     count = 1
        self.val = val
        self.sum += val 
        self.count += count
        self.avg = self.sum / self.count

class MetricMeter(object):
    def __init__(self, metrics=None):
        self.meters = []
        if metrics is not None:
            for k in metrics:
                self.set_meter(k)
    
    def set_meter(self, name):
        if hasattr(self, name): return
        
        self.meters.append(name)
        self.__setattr__(name, AverageValueMeter())

    def update(self, metric_dict: dict):
        for m, v in metric_dict.items():
            if not hasattr(self, m):
                self.set_meter(m)
            if isinstance(v, dict):
                self.__getattribute__(m).update(**v)
            else:
                self.__getattribute__(m).update(v)
  
    def reset(self):
        for m in self.meters:
            self.__getattribute__(m).reset()
    
    def info(self):
        log = ''
        for m in self.meters:
            log += '{}: {:.6f}, '.format(m, self.__getattribute__(m).avg)
        return log





     


    