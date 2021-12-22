import tensorflow as tf
import tfsde

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
        sys.path.append(module_path)



os.environ["CUDA_VISIBLE_DEVICES"]="1"

class SDE(tf.Module):

    def __init__(self):
        super().__init__()
        self.theta = tf.Variable(0.1, trainable=True)  # Scalar parameter.
        self.noise_type = "diagonal"
        self.sde_type = "ito"
                                                
    def f(self, t, y):
        return tf.sin(t) + self.theta * y
                                                                
    def g(self, t, y):
        return 0.3 * tf.sigmoid(tf.cos(t) * tf.exp(-y))


if __name__ == "__main__":

    batch_size, state_size, t_size = 3, 1, 100
    sde = SDE()
    ts = tf.linspace(0.0, 1.0, t_size)
    y0 = tf.fill(dims=(batch_size, state_size), value=0.1)

    #ys = tf.stop_gradient(tfsde.sdeint(sde, y0, ts, method='euler'))


    bm = tfsde.BrownianInterval(t0=0.0, t1=1.0, size=(batch_size, state_size))
    #print(bm(0.0, 0.5))

    #bm_increments = tf.stack([bm(t0, t1) for t0, t1 in zip(ts[:-1], ts[1:])], axis=0)
    #bm_queries = tf.concat([tf.zeros([1, batch_size, state_size]), tf.math.cumsum(bm_increments, axis=0)], axis=0)


    bm_increments2 = tf.stack([bm(t0, t1) for t0, t1 in zip(ts[:-1], ts[1:])], axis=0)
    bm_queries2 = tf.concat([tf.zeros([1, batch_size, state_size]), tf.math.cumsum(bm_increments2, axis=0)], axis=0)


    #if torch.cuda.is_available():
    #    gpu = torch.device('cuda')
    #    sde = SDE().to(gpu)
    #    ts = ts.to(gpu)
    #    y0 = y0.to(gpu)
    #    with torch.no_grad():
    #        ys = torchsde.sdeint(sde, y0, ts, method='euler')  # (100, 3, 1).

    #bm = torchsde.BrownianInterval(t0=0.0, t1=1.0, size=(batch_size, state_size), device='cuda')

    #ys = torchsde.sdeint(sde, y0, ts, method='euler', bm=bm)
    #y_final = ys[-1]
    #target = torch.randn_like(y_final)
    #loss = ((target - y_final) ** 2).sum(dim=1).mean(dim=0)
    #loss.backward()
    #print(sde.theta.grad)
    #
    #ys = torchsde.sdeint_adjoint(sde, y0, ts, method='euler', bm=bm)
    #y_final = ys[-1]
    #target = torch.randn_like(y_final)
    #loss = ((target - y_final) ** 2).sum(dim=1).mean(dim=0)
    #loss.backward()
    #print(sde.theta.grad)
