"""Gera o vaso de orquídea ripado (paramétrico) e exporta em STL.

Recria o vaso_fundo_redondo_esp5mm.stl (mesmas medidas principais) como uma
malha única e fechada, e acrescenta:
  - rugas (frisos em dente de serra) no lado interno das ripas, para as
    raízes da orquídea se prenderem;
  - anel reforçado no topo e na base das paredes laterais;
  - pezinhos radiais embaixo, que levantam o vaso e deixam a água escorrer.

Uso:  pip install manifold3d trimesh numpy
      python3 gerar_vaso.py            -> vaso_orquidea_rugas_pes.stl
Todas as medidas em mm. O corpo do vaso começa em z = 0 (face de baixo do
fundo); os pés ficam abaixo disso e o modelo é transladado para apoiar em z = 0.
"""

import math

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold, OpType

# ---- Corpo (medidas do vaso original) ---------------------------------------
ALTURA = 91.0          # altura do corpo (fundo + paredes)
R_BASE_INT = 43.0      # raio interno da parede em z = 0
R_TOPO_INT = 93.0      # raio interno da parede em z = 90
ESP_PAREDE = 2.0       # espessura radial das ripas
ESP_FUNDO = 5.0        # espessura do fundo

# ---- Ripas ------------------------------------------------------------------
N_RIPAS = 96
LARG_RIPA_BASE = 1.6   # largura tangencial embaixo
LARG_RIPA_TOPO = 2.4   # largura tangencial em cima

# ---- Rugas internas (anéis em dente de serra) -------------------------------
RUGA_Z_INI = 14.0
RUGA_Z_FIM = 80.0
RUGA_PASSO = 8.0
RUGA_SALIENCIA = 1.5   # quanto a ruga avança para dentro do vaso
RUGA_ALTURA = 3.0      # altura da face inferior (inclinada, imprime sem suporte)

# ---- Anéis das paredes ------------------------------------------------------
ANEL_TOPO_ALT = 5.0
ANEL_TOPO_R_EXT = 96.0
ANEL_BASE_ALT = 7.0
ANEL_BASE_R_EXT = 50.5

# ---- Cone central de aeração ------------------------------------------------
N_HASTES = 12
HASTE_R_BASE = 30.0
HASTE_R_TOPO = 5.0
HASTE_ALT = 60.0

# ---- Pés --------------------------------------------------------------------
N_PES = 10
PE_ALTURA = 5.0
PE_LARGURA = 4.0
PE_R_EXT = 50.0

# ---- Furos de drenagem no fundo ---------------------------------------------
FURO_D = 6.0

SEG = 256  # segmentos de círculo


def r_int(z):
    """Raio interno da parede na altura z."""
    return R_BASE_INT + (R_TOPO_INT - R_BASE_INT) * z / 90.0


def r_ext(z):
    return r_int(z) + ESP_PAREDE


def revolve(pontos_rz):
    """Gira o perfil (r, z) em torno do eixo Z."""
    p = np.array(pontos_rz)
    area = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area < 0:  # CrossSection espera contorno anti-horário
        p = p[::-1]
    return CrossSection([p.tolist()]).revolve(circular_segments=SEG)


def bloco_inclinado(ang, r0, r1, larg0, z0, r2, r3, larg1, z1):
    """Hull de duas seções retangulares radiais (embaixo e em cima)."""
    u = np.array([math.cos(ang), math.sin(ang)])
    t = np.array([-math.sin(ang), math.cos(ang)])
    pts = []
    for ra, rb, w, z in ((r0, r1, larg0, z0), (r2, r3, larg1, z1)):
        for r in (ra, rb):
            for s in (-0.5, 0.5):
                xy = u * r + t * w * s
                pts.append([xy[0], xy[1], z])
    return Manifold.hull_points(pts)


def ripas():
    # As pontas ficam embutidas no fundo e no anel do topo (evita faces
    # coplanares, que geram lascas na malha).
    z0, z1 = 1.0, ALTURA - 1.0
    return [
        bloco_inclinado(2 * math.pi * i / N_RIPAS,
                        r_int(z0), r_ext(z0), LARG_RIPA_BASE, z0,
                        r_int(z1), r_ext(z1), LARG_RIPA_TOPO, z1)
        for i in range(N_RIPAS)
    ]


def rugas():
    """Anéis internos em dente de serra: face de baixo quase vertical
    (acompanha a parede, sem balanço) e face de cima em degrau, onde as
    raízes se agarram. Os anéis também amarram as ripas entre si."""
    out = []
    z = RUGA_Z_INI
    while z <= RUGA_Z_FIM + 1e-6:
        za, zb = z - RUGA_ALTURA, z + 1.0
        out.append(revolve([
            (r_int(za) + 1.0, za),               # dentro da ripa
            (r_int(za) + 0.2, za),
            (r_int(z) - RUGA_SALIENCIA, z),      # crista da ruga
            (r_int(zb) + 0.2, zb),
            (r_int(zb) + 1.0, zb),
        ]))
        z += RUGA_PASSO
    return out


def anel_topo():
    z0 = ALTURA - ANEL_TOPO_ALT
    ri = r_int(z0) - 0.5
    return revolve([
        (ri, z0),
        (r_ext(z0), z0),
        (ANEL_TOPO_R_EXT, z0 + 3.5),   # chanfro externo ~45°, sem suporte
        (ANEL_TOPO_R_EXT, ALTURA),
        (ri, ALTURA),
    ])


def fundo_e_anel_base():
    """Fundo (disco) + anel na base das paredes, em uma peça só."""
    ri = r_int(0) - 1.0
    return revolve([
        (0.0, 0.0),
        (ANEL_BASE_R_EXT, 0.0),
        (ANEL_BASE_R_EXT, ANEL_BASE_ALT),
        (ri, ANEL_BASE_ALT),
        (ri, ESP_FUNDO),
        (0.0, ESP_FUNDO),
    ])


def cone_central():
    hastes = [
        bloco_inclinado(2 * math.pi * (i + 0.5) / N_HASTES,
                        HASTE_R_BASE - 2.0, HASTE_R_BASE, 5.0, 1.0,
                        HASTE_R_TOPO - 2.0, HASTE_R_TOPO, 1.5, HASTE_ALT)
        for i in range(N_HASTES)
    ]
    tampa = Manifold.cylinder(2.0, HASTE_R_TOPO + 1.0, HASTE_R_TOPO + 1.0,
                              64).translate((0, 0, HASTE_ALT - 2.0))
    return hastes + [tampa]


def ang_pe(i):
    return 2 * math.pi * i / N_PES


def pes():
    """Pés em raios (como uma roda): a água escorre pelos canais entre eles."""
    out = []
    for i in range(N_PES):
        p = Manifold.cube((PE_R_EXT, PE_LARGURA, PE_ALTURA + 0.5)) \
            .translate((0, -PE_LARGURA / 2, -PE_ALTURA))
        out.append(p.rotate((0, 0, math.degrees(ang_pe(i)))))
    out.append(Manifold.cylinder(PE_ALTURA + 0.5, 6.0, 6.0, 64)
               .translate((0, 0, -PE_ALTURA)))
    return out


def furos():
    """Furos de drenagem entre os pés (caem direto nos canais)."""
    out = []
    furo = Manifold.cylinder(ESP_FUNDO + 2, FURO_D / 2, FURO_D / 2, 32) \
        .translate((0, 0, -1))
    meio = math.pi / N_PES
    for i in range(N_PES):
        a = ang_pe(i) + meio
        posicoes = [(13.0, a), (23.0, a), (37.0, a - meio * 0.4), (37.0, a + meio * 0.4)]
        for r, ang in posicoes:
            out.append(furo.translate((r * math.cos(ang), r * math.sin(ang), 0)))
    return out


def gerar():
    corpo = Manifold.batch_boolean(
        [fundo_e_anel_base(), anel_topo()] + ripas() + rugas()
        + cone_central() + pes(),
        OpType.Add)
    corpo = corpo - Manifold.batch_boolean(furos(), OpType.Add)
    return corpo.translate((0, 0, PE_ALTURA))


def main(saida="vaso_orquidea_rugas_pes.stl"):
    vaso = gerar()
    m = vaso.to_mesh()
    malha = trimesh.Trimesh(vertices=m.vert_properties[:, :3], faces=m.tri_verts)
    malha.export(saida)
    print(f"{saida}: {len(malha.faces)} triângulos, "
          f"fechada={malha.is_watertight}, volume={malha.volume / 1000:.1f} cm³, "
          f"dimensões={np.round(malha.extents, 1).tolist()} mm")


if __name__ == "__main__":
    main()
