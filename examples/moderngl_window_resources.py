import os
import glm
import moderngl
import moderngl_window as mglw
import numpy as np


class Example(mglw.WindowConfig):
    title = 'ModernGL Orthographic Camera Example'
    gl_version = (3, 3)
    window_size = (800, 800)
    aspect_ratio = 1.0
    resizable = True

    resource_dir = os.path.normpath(os.path.join(__file__, '../data'))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 加载模型和纹理
        self.obj = self.load_scene('models/sitting_dummy.obj')
        self.texture = self.load_texture_2d('textures/wood.jpg')

        # 着色器程序
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;
                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;
                void main() {
                    v_vert = in_position;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                    gl_Position = Mvp * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Texture;
                uniform vec4 Color;
                uniform vec3 Light;
                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;
                out vec4 f_color;
                void main() {
                    float lum = dot(normalize(v_norm), normalize(v_vert - Light));
                    lum = acos(lum) / 3.14159265;
                    lum = clamp(lum, 0.0, 1.0);
                    lum = lum * lum;
                    lum = smoothstep(0.0, 1.0, lum);
                    lum *= smoothstep(0.0, 80.0, v_vert.z) * 0.3 + 0.7;
                    lum = lum * 0.8 + 0.2;
                    vec3 color = texture(Texture, v_text).rgb;
                    color = color * (1.0 - Color.a) + Color.rgb * Color.a;
                    f_color = vec4(color * lum, 1.0);
                }
            '''
        )

        self.vao = self.obj.root_nodes[0].mesh.vao.instance(self.program)
        all_vertices = []
        vertex_data = None
        for mesh in self.obj.meshes:
            # 每个 mesh 通常有一个 vbo，包含所有顶点位置
            vbo = mesh.vao._buffers[0].buffer
            # 注意：这里拿到的是 ModernGL buffer，需要读取出数据
            vertex_data = np.frombuffer(vbo.read(), dtype='f4')  # float32
            vertex_data = vertex_data.reshape((-1, 3))  # 3个一组，x/y/z
            
            all_vertices.append(vertex_data)

        if all_vertices:
            all_vertices = np.vstack(all_vertices)  # 合并所有
            # 按列分别求最小值最大值，得到 bbox
            bbox_min = np.min(all_vertices, axis=0)  # [min_x, min_y, min_z]
            bbox_max = np.max(all_vertices, axis=0)  # [max_x, max_y, max_z]
            
            self.orbit_center = (bbox_min + bbox_max) * 0.5
            # self.orbit_center = bbox_max
        # max_point_size = self.ctx.point_size()
        # print(f"Maximum point size: {max_point_size}")


        # 点精灵用的单独 program
        self.point_program = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    gl_PointSize = 10.0; // 点大小
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 fragColor;
                void main() {
                    vec2 center = gl_PointCoord - vec2(0.5, 0.5);  // 从中心 (0.5, 0.5) 开始

                    float dist = length(center);

                    if (dist > 0.5) {
                        discard;
                    }
                    fragColor = vec4(1.0, 0.0, 0.0, 1-dist);  // 根据距离设置透明度，创建渐变效果
                }
            '''
        )

        # 点精灵的 VAO，只有1个顶点（orbit_center位置）
        self.point_vbo = self.ctx.buffer(reserve=12)  # 3个float，占12字节
        self.point_vao = self.ctx.simple_vertex_array(self.point_program, self.point_vbo, 'in_position')
        # self.ctx.gc_mode()

        # 相机控制参数
        self.camera_distance = 300.0
        self.camera_rotation = glm.vec2(0.0, 0.0)  # yaw, pitch
        self.camera_pan = glm.vec2(0.0, 0.0)

        # 模型中心（用于围绕旋转）
        # self.orbit_center = (self.obj.bbox_min + self.obj.bbox_max) * 0.5

        self.model_matrix = glm.mat4(1.0)
        self.drag_mode = None

        self.depth_texture = self.ctx.depth_texture(self.window_size)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture(self.window_size, 4)], depth_attachment=self.depth_texture)

    
    def reset_orbit_center(self, new_center):
    # 记录原始的 view 矩阵
        original_view = self.view

        # 更新 orbit_center
        self.orbit_center = new_center

        # 计算平移差值，确保 view 矩阵一致
        # 获取原始的平移向量
        original_translation = original_view[3]
        self.camera_matrix()
        new_translation = self.view[3]

        # 计算新的平移差值，并更新 camera_pan
        translation_diff = original_translation - new_translation
        self.camera_pan.x += translation_diff[0]
        self.camera_pan.y += translation_diff[1]

        self.camera_matrix()

        print(f"Original view matrix: {original_view}")
        print(f"Updated view matrix: {self.view}")
        print(f"Translation diff: {translation_diff}")
        print(f"Updated camera pan: {self.camera_pan}")


    def camera_matrix(self):
        self.yaw, self.pitch = self.camera_rotation
        pan_x, pan_y = self.camera_pan

        self.view = glm.mat4(1.0)
        # self.view = glm.rotate(self.view, glm.radians(pitch), glm.vec3(1, 0, 0))  # 绕X轴旋转
        # self.view = glm.rotate(self.view, glm.radians(yaw), glm.vec3(0, 0, 1))    # 绕Z轴旋转

        # 视图矩阵（围绕 orbit_center 旋转 + 平移 + 拉远）
        self.view = glm.translate(self.view, glm.vec3(pan_x, pan_y, 0.0))
        self.view = glm.translate(self.view, self.orbit_center)
        self.view = glm.rotate(self.view, glm.radians(self.pitch), glm.vec3(1, 0, 0))
        self.view = glm.rotate(self.view, glm.radians(self.yaw), glm.vec3(0, 0, 1))
        self.view = glm.translate(self.view, -self.orbit_center)

        # 正交投影矩阵
        # 确保相机距离在合理范围内
        scale = max(self.camera_distance, 0.1) * 0.01  # 防止相机距离过小
        left = -self.aspect_ratio * 100 * scale
        right = self.aspect_ratio * 100 * scale
        bottom = -100 * scale
        top = 100 * scale
        near = -1000.0
        far = 1000.0

        self.proj = glm.ortho(left, right, bottom, top, near, far)
        return self.proj * self.view

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self.drag_mode == 'rotate':
            self.camera_rotation.x += dx * 0.5
            self.camera_rotation.y += dy * 0.5
            # self.camera_rotation.y = glm.clamp(self.camera_rotation.y, -180.0, 180.0)

        elif self.drag_mode == 'pan':
            # scale = self.camera_distance * 0.003  # 平移比例，跟随缩放远近
            # self.camera_pan.x += dx * scale
            # self.camera_pan.y -= dy * scale
            ortho_scale = self.camera_distance
            world_dx = dx / self.wnd.buffer_width * ortho_scale * 2
            world_dy = dy / self.wnd.buffer_height * ortho_scale * 2 / self.aspect_ratio

            self.camera_pan.x += world_dx
            self.camera_pan.y -= world_dy

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.camera_distance -= y_offset * 10.0
        self.camera_distance = glm.clamp(self.camera_distance, 50.0, 1000.0)

    def on_mouse_press_event(self, x, y, button):
        if button == 3:  # 中键
            flipped_y = self.wnd.buffer_height - y

            # 读取整张深度纹理（默认是 float32 格式，不需要传dtype）
            depth_bytes = self.depth_texture.read()

            depth_data = np.frombuffer(depth_bytes, dtype=np.float32)
            depth_data = depth_data.reshape((self.depth_texture.height, self.depth_texture.width))

            if 0 <= x < self.depth_texture.width and 0 <= flipped_y < self.depth_texture.height:
                depth_value = depth_data[int(flipped_y), int(x)]
                print(f"鼠标点击屏幕({x},{y}) 深度值：{depth_value}")

                # 反推 3D 世界坐标
                viewport_width = self.wnd.buffer_width
                viewport_height = self.wnd.buffer_height

                clip_x = (2.0 * x) / viewport_width - 1.0
                clip_y = (2.0 * flipped_y) / viewport_height - 1.0  # 注意是flipped_y
                clip_z = 2.0 * depth_value - 1.0

                clip_coord = np.array([clip_x, clip_y, clip_z, 1.0], dtype=np.float32)

                inv_proj = np.linalg.inv(self.proj)
                inv_view = np.linalg.inv(self.view)

                view_coord = inv_proj @ clip_coord
                view_coord /= view_coord[3]

                world_coord = inv_view @ view_coord
                world_coord /= world_coord[3]

                # self.reset_orbit_center(world_coord[:3])
                # print("View Matrix:", self.view)  
                # # 更新 orbit_center
                # self.orbit_center = world_coord[:3]
                self.reset_orbit_center(world_coord[:3])
                # self.camera_pan.x = 0
                # self.camera_pan.y = 0
                # print(f"新的orbit_center设置为: {self.orbit_center}")

                # 确保视图矩阵和投影矩阵正确更新
                self.camera_matrix()  # 更新视图矩阵
                # print("View Matrix:", self.view)  
            else:
                print("点击超出范围")

        if button == 1:
            self.drag_mode = 'rotate'
        elif button == 2:
            self.drag_mode = 'pan'

    def on_mouse_release_event(self, x, y, button):
        self.drag_mode = None

    def on_render(self, time, frame_time):
        self.fbo.use()  # <-- 渲染到 Framebuffer 而不是默认屏幕
        self.fbo.clear(1.0, 1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.program['Light'].value = (-140.0, -300.0, 350.0)
        self.program['Color'].value = (1.0, 1.0, 1.0, 0.25)

        mvp = self.camera_matrix() * self.model_matrix
        self.program['Mvp'].write(mvp)

        self.texture.use()
        self.vao.render()

        # 点精灵
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.DEFAULT_BLENDING

        # 更新点精灵位置
        # 使用 numpy 数组替代 .astype('f4')
        self.point_vbo.write(np.array(self.orbit_center).astype('f4').tobytes())

        self.point_program['Mvp'].write(mvp)
        self.point_vao.render(mode=moderngl.POINTS)

        # 渲染完后，再切换回默认 framebuffer，画最终画面（可选）
        self.ctx.screen.use()
        self.ctx.copy_framebuffer(self.ctx.screen, self.fbo)


if __name__ == '__main__':
    Example.run()
