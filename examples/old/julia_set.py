import numpy as np

import moderngl
from _example import Example


class Fractal(Example):
    title = "Julia Set"
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                in vec2 in_vert;
                out vec2 v_text;

                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_text = in_vert;
                }
            ''',
            fragment_shader='''
                #version 330

                in vec2 v_text;
                out vec4 f_color;

                uniform sampler2D Texture;
                uniform vec2 Seed;
                uniform int Iter;

                void main() {
                    vec2 c = Seed;
                    int i;

                    vec2 z = vec2(3.0 * v_text.x, 2.0 * v_text.y);

                    for (i = 0; i < Iter; i++) {
                        float x = (z.x * z.x - z.y * z.y) + c.x;
                        float y = (z.y * z.x + z.x * z.y) + c.y;

                        if ((x * x + y * y) > 4.0) {
                            break;
                        }

                        z.x = x;
                        z.y = y;
                    }

                    f_color = texture(Texture, vec2((i == Iter ? 0.0 : float(i)) / 100.0, 0.0));
                }
            ''',
        )

        self.seed = self.prog['Seed']
        self.iter = self.prog['Iter']

        self.texture = self.load_texture_2d('pal.png')

        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])

        self.vbo = self.ctx.buffer(vertices.astype('f4'))
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

    def on_render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)

        self.seed.value = (-0.8, 0.156)
        self.iter.value = 100

        self.texture.use()
        self.vao.render(moderngl.TRIANGLE_STRIP)


if __name__ == '__main__':
    Fractal.run()
