# ref: https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow

# Migrating tf1 to tf2 : https://www.tensorflow.org/guide/migrate

#import tensorflow as tf

# ALTENATIVELY, if use tf2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def scoping(fn, scope1, scope2, vals):
    with fn(scope1):
        a = tf.Variable(vals[0], name='a')
        b = tf.get_variable('b', initializer=vals[1])
        c = tf.constant(vals[2], name='c')

        with fn(scope2):
            d = tf.add(a * b, c, name='res')

        print ('\n  '.join([scope1, a.name, b.name, c.name, d.name]), '\n')
    return d

d1 = scoping(tf.variable_scope, 'scope_vars', 'res', [1, 2, 3])
d2 = scoping(tf.name_scope,     'scope_name', 'res', [1, 2, 3])

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs', sess.graph)
    sess.run(tf.global_variables_initializer())
    print (sess.run([d1, d2]))
    writer.close()
