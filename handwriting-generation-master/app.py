import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
import seaborn as sns
from collections import namedtuple
import streamlit as st

# Set page config for a premium, wide layout
st.set_page_config(
    page_title="Handwriting Generation Dashboard",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling (glassmorphism, clean fonts, card styling)
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #6d6d6d;
        margin-bottom: 2rem;
    }
    .developer-card {
        background-color: rgba(255, 75, 75, 0.05);
        border: 1px solid rgba(255, 75, 75, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 2rem;
    }
    .dev-title {
        font-weight: 700;
        color: #FF4B4B;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions from original generate.py
def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0):
    Xmu = X - mux
    Ymu = Y - muy
    if sigmaxy == 0.0:
        return np.exp(-0.5 * (Xmu**2 / sigmax**2 + Ymu**2 / sigmay**2)) / (2.0 * np.pi * sigmax * sigmay)
    else:
        rho = sigmaxy / (sigmax * sigmay)
        z = Xmu**2 / sigmax**2 + Ymu**2 / sigmay**2 - 2.0 * rho * Xmu * Ymu / (sigmax * sigmay)
        denom = 2.0 * np.pi * sigmax * sigmay * np.sqrt(1.0 - rho**2)
        return np.exp(-z / (2.0 * (1.0 - rho**2))) / denom

def sample(e, mu1, mu2, std1, std2, rho):
    cov = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])
    x, y = np.random.multivariate_normal(mean, cov)
    end = np.random.binomial(1, e)
    return np.array([x, y, end])

def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    return strokes

def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)

# Cache model loading to avoid reloading on every generation
@st.cache_resource
def load_tf_session(model_path):
    graph = tf.Graph()
    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(graph=graph, config=config)
    with graph.as_default():
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)
    return sess, graph

@st.cache_data
def load_data_files():
    with open(os.path.join('data', 'translation.pkl'), 'rb') as file:
        translation = pickle.load(file)
    with open(os.path.join('data', 'styles.pkl'), 'rb') as file:
        styles = pickle.load(file)
    return translation, styles

# Generation logic updated to accept bias dynamically
def sample_text_streamlit(sess, graph, args_text, translation, bias_value, style=None):
    with graph.as_default():
        fields = ['coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2',
                  'rho', 'window', 'kappa', 'phi', 'finish', 'zero_states']
        vs = namedtuple('Params', fields)(
            *[tf.get_collection(name)[0] for name in fields]
        )

        text = np.array([translation.get(c, 0) for c in args_text])
        coord = np.array([0., 0., 1.])
        coords = [coord]

        # Prime the model with the style if requested
        prime_len, style_len = 0, 0
        if style is not None:
            style_coords, style_text = style
            prime_len = len(style_coords)
            style_len = len(style_text)
            prime_coords = list(style_coords)
            coord = prime_coords[0]
            text = np.r_[style_text, text]
            sequence_prime = np.eye(len(translation), dtype=np.float32)[style_text]
            sequence_prime = np.expand_dims(np.concatenate([sequence_prime, np.zeros((1, len(translation)))]), axis=0)

        sequence = np.eye(len(translation), dtype=np.float32)[text]
        sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

        phi_data, window_data, kappa_data, stroke_data = [], [], [], []
        sess.run(vs.zero_states)
        sequence_len = len(args_text) + style_len
        
        # Progress bar setup inside Streamlit
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        total_steps = 60 * sequence_len + 1
        for s in range(1, total_steps):
            is_priming = s < prime_len

            # Update progress indicator periodically to keep UI smooth
            if s % 10 == 0 or s == total_steps - 1:
                progress_percent = int((s / total_steps) * 100)
                progress_bar.progress(progress_percent)
                progress_text.text(f"Generating handwriting... Step {s}/{total_steps} ({'priming' if is_priming else 'synthesis'})")

            e, pi, mu1, mu2, std1, std2, rho, \
            finish, phi, window, kappa = sess.run([vs.e, vs.pi, vs.mu1, vs.mu2,
                                                   vs.std1, vs.std2, vs.rho, vs.finish,
                                                   vs.phi, vs.window, vs.kappa],
                                                  feed_dict={
                                                      vs.coordinates: coord[None, None, ...],
                                                      vs.sequence: sequence_prime if is_priming else sequence,
                                                      vs.bias: bias_value
                                                  })

            if is_priming:
                coord = prime_coords[s]
            else:
                phi_data += [phi[0, :]]
                window_data += [window[0, :]]
                kappa_data += [kappa[0, :]]
                g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
                coord = sample(e[0, 0], mu1[0, g], mu2[0, g],
                               std1[0, g], std2[0, g], rho[0, g])
                coords += [coord]
                stroke_data += [[mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

                if finish[0, 0] > 0.8:
                    break

        progress_bar.empty()
        progress_text.empty()
        coords = np.array(coords)
        coords[-1, 2] = 1.

        return phi_data, window_data, kappa_data, stroke_data, coords

def main():
    st.markdown('<div class="main-title">✍️ Handwriting Generation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Generate realistic handwriting samples from text using recurrent neural networks (Graves paper implementation).</div>', unsafe_allow_html=True)

    # Initialize model session and configs
    model_path = os.path.join('pretrained', 'model-29')
    
    with st.spinner("Initializing TensorFlow model (this may take a few seconds on first run)..."):
        sess, graph = load_tf_session(model_path)
        translation, styles = load_data_files()

    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''

    # Sidebar configuration
    st.sidebar.header("🔧 Settings")
    
    # Style Selector
    style_idx = st.sidebar.selectbox(
        "Handwriting Style",
        options=list(range(len(styles[0]))),
        format_func=lambda x: f"Style {x}",
        index=2
    )

    # Bias Selector
    bias_value = st.sidebar.slider(
        "Bias (Higher = clearer writing)",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    # Debug Details checkbox
    show_details = st.sidebar.checkbox("Show model internal state plots (phi, attention window)", value=True)

    # Developer profile card in sidebar
    st.sidebar.markdown(f"""
    <div class="developer-card">
        <div class="dev-title">👨‍💻 Developer</div>
        <strong>Chhotelal Kushwaha</strong><br>
        <small>B.Tech Computer Science & Engineering</small><br>
        <small>HNBGU University</small>
        <hr style="margin: 0.8rem 0; border: 0; border-top: 1px solid rgba(0,0,0,0.1)">
        <a href="https://github.com/karan2027" target="_blank">🔗 GitHub Profile</a><br>
        <a href="https://linkedin.com/in/chhotelal-kushwaha-2902a3329" target="_blank">🔗 LinkedIn</a><br>
        <a href="mailto:chhotelalkushwahak9628@gmail.com">📧 Email Dev</a>
    </div>
    """, unsafe_allow_html=True)

    # Main layout area
    text_input = st.text_input("Enter text to generate (only letters, numbers, and common symbols):", "Handwriting generation by Chhotelal")
    
    if st.button("✍️ Generate Handwriting", type="primary"):
        if not text_input:
            st.error("Please enter some text to write.")
            return

        with st.spinner("Generating..."):
            style = [styles[0][style_idx], styles[1][style_idx]]
            phi_data, window_data, kappa_data, stroke_data, coords = sample_text_streamlit(
                sess, graph, text_input, translation, bias_value, style
            )

        # Plot result
        st.subheader("Generated Handwriting Output")
        
        strokes = np.array(stroke_data)
        epsilon = 1e-8
        strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
        minx, maxx = np.min(strokes[:, 0]), np.max(strokes[:, 0])
        miny, maxy = np.min(strokes[:, 1]), np.max(strokes[:, 1])

        if show_details:
            delta = abs(maxx - minx) / 400.
            # Handle edge case where delta is too small or zero
            if delta < 1e-3:
                delta = 1e-3
            x = np.arange(minx, maxx, delta)
            y = np.arange(miny, maxy, delta)
            x_grid, y_grid = np.meshgrid(x, y)
            z_grid = np.zeros_like(x_grid)
            for i in range(strokes.shape[0]):
                gauss = bivariate_normal(x_grid, y_grid, mux=strokes[i, 0], muy=strokes[i, 1],
                                              sigmax=strokes[i, 2], sigmay=strokes[i, 3],
                                              sigmaxy=0.)
                z_grid += gauss * np.power(strokes[i, 2] + strokes[i, 3], 0.4) / (np.max(gauss) + epsilon)

            fig, ax = plt.subplots(2, 2, figsize=(14, 10))

            # Densities
            ax[0, 0].imshow(z_grid, interpolation='bilinear', aspect='auto', cmap=cm.jet)
            ax[0, 0].grid(False)
            ax[0, 0].set_title('Densities')
            ax[0, 0].set_aspect('equal')

            # Handwriting Plot
            for stroke in split_strokes(cumsum(np.array(coords))):
                ax[0, 1].plot(stroke[:, 0], -stroke[:, 1], color='#FF4B4B', linewidth=2)
            ax[0, 1].set_title('Handwriting')
            ax[0, 1].set_aspect('equal')

            # Phi attention window alignment
            phi_img = np.vstack(phi_data).T[::-1, :]
            ax[1, 0].imshow(phi_img, interpolation='nearest', aspect='auto', cmap=cm.jet)
            ax[1, 0].set_yticks(np.arange(0, len(text_input) + 1))
            ax[1, 0].set_yticklabels(list(' ' + text_input[::-1]), rotation='vertical', fontsize=8)
            ax[1, 0].grid(False)
            ax[1, 0].set_title('Phi (Alignment Matrix)')

            # Window
            window_img = np.vstack(window_data).T
            ax[1, 1].imshow(window_img, interpolation='nearest', aspect='auto', cmap=cm.jet)
            ax[1, 1].set_yticks(np.arange(0, len(charset)))
            ax[1, 1].set_yticklabels(list(charset), rotation='vertical', fontsize=8)
            ax[1, 1].grid(False)
            ax[1, 1].set_title('Window (Character attention)')

            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            for stroke in split_strokes(cumsum(np.array(coords))):
                ax.plot(stroke[:, 0], -stroke[:, 1], color='#FF4B4B', linewidth=2.5)
            ax.set_title('Handwriting', fontsize=14)
            ax.set_aspect('equal')
            ax.axis('off')
            st.pyplot(fig)

if __name__ == '__main__':
    main()
