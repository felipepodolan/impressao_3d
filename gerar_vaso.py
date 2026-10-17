"""Gera o vaso de orquídea ripado (paramétrico) e exporta em STL.

Recria o vaso_fundo_redondo_esp5mm.stl (mesmas medidas principais) como uma
malha única e fechada, e acrescenta:
  - rugas (frisos em dente de serra) no lado interno das ripas, para as
    raízes da orquídea se prenderem;
  - anel reforçado no topo e na base das paredes laterais;
  - pezinhos radiais embaixo, que levantam o vaso e deixam a água escorrer;
  - pontes (peças separadas, 3 tamanhos) que se apoiam no anel do topo e têm
    furos ao longo de todo o comprimento para posicionar os arames da haste.

Uso:  pip install manifold3d trimesh numpy
      python3 gerar_vaso.py   -> vaso_orquidea_rugas_pes.stl
                                 ponte_arames_{G,M,P}.stl
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

# ---- Pontes para os arames da haste (peças separadas) -----------------------
# Cada ponte é uma barra que atravessa o vaso de um lado a outro do anel do
# topo (uma corda do círculo) e fica acima do substrato. "Afastamento" é a
# distância da ponte até o centro do vaso: 0 = passa pelo meio (diâmetro).
PONTES = (("G", 0.0), ("M", 40.0), ("P", 65.0))
PONTE_LARG = 8.0       # largura da barra
PONTE_ALT = 6.0        # altura da barra
PONTE_FURO_D = 3.4     # furos para o arame (até ~3 mm)
PONTE_PASSO = 7.0      # distância entre furos
PONTE_FOLGA = 0.3      # folga dos encaixes em relação ao anel
PONTE_ESP = 2.0        # espessura das paredes dos encaixes

SEG = 256  # segmentos de círculo


def r_int(z):
    """Raio interno da parede na altura z."""
    return R_BASE_INT + (R_TOPO_INT - R_BASE_INT) * z / 90.0


def r_ext(z):
    return r_int(z) + ESP_PAREDE


def revolve(pontos_rz, graus=360.0):
    """Gira o perfil (r, z) em torno do eixo Z."""
    p = np.array(pontos_rz)
    area = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area < 0:  # CrossSection espera contorno anti-horário
        p = p[::-1]
    return CrossSection([p.tolist()]).revolve(circular_segments=SEG,
                                              revolve_degrees=graus)


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


def ponte(afastamento, posicao_impressao=True):
    """Barra reta com furos para arame ao longo de todo o comprimento e, em
    cada ponta, um encaixe em U invertido que se apoia por cima do anel do
    topo (uma perna por dentro, outra por fora). Ela só é apoiada: dá para
    deslizar a ponte ao longo do anel e escolher a posição.
    Sai na posição de impressão (face de cima na mesa, sem suporte)."""
    z0 = ALTURA - ANEL_TOPO_ALT
    ri = r_int(z0) - 0.5                       # face interna do anel
    c, t = PONTE_FOLGA, PONTE_ESP
    z_top = ALTURA + c                         # face de baixo da ponte
    z_sup = z_top + t                          # face de cima
    r_perna = ri - c                           # face da perna interna
    meia_corda = math.sqrt(r_perna ** 2 - afastamento ** 2)
    meio_arco = (PONTE_LARG / 2 + t + 2.0) / ri       # rad, meia largura do U

    # Tampo: face de cima inteira numa peça só (barra + topo dos dois U),
    # para não haver faces coplanares entre as partes.
    def arco(r0, r1, centro):
        angs = np.linspace(centro - meio_arco, centro + meio_arco, 24)
        ext = [(r1 * math.cos(a), r1 * math.sin(a)) for a in angs]
        int_ = [(r0 * math.cos(a), r0 * math.sin(a)) for a in angs[::-1]]
        return CrossSection([ext + int_])

    angs_pontas = [math.atan2(lado * meia_corda, afastamento) for lado in (1, -1)]
    barra2d = CrossSection.square((PONTE_LARG, 2 * meia_corda + 10)) \
        .translate((afastamento - PONTE_LARG / 2, -meia_corda - 5)) \
        ^ CrossSection.circle(r_perna - 0.5, SEG)
    tampo2d = barra2d
    for ang in angs_pontas:
        tampo2d = tampo2d + arco(r_perna - t, ANEL_TOPO_R_EXT + c + t, ang)
    pecas = [tampo2d.extrude(t).translate((0, 0, z_top))]

    # corpo da barra, embutido 0,5 mm no tampo
    pecas.append(barra2d.extrude(PONTE_ALT - t + 0.5)
                 .translate((0, 0, z_sup - PONTE_ALT)))

    # pernas do U: interna (desce por dentro do anel) e aba externa
    pernas = [
        [(r_perna - t, z0 - 1.0), (r_perna, z0 - 1.0),
         (r_perna, z_top + 0.5), (r_perna - t, z_top + 0.5)],
        [(ANEL_TOPO_R_EXT + c, z0 + 3.5), (ANEL_TOPO_R_EXT + c + t, z0 + 3.5),
         (ANEL_TOPO_R_EXT + c + t, z_top + 0.5), (ANEL_TOPO_R_EXT + c, z_top + 0.5)],
    ]
    graus = math.degrees(2 * meio_arco - 0.6 / ri)  # um pouco mais estreitas
    for ang in angs_pontas:
        for perfil in pernas:
            pecas.append(revolve(perfil, graus)
                         .rotate((0, 0, math.degrees(ang) - graus / 2)))
    peca = Manifold.batch_boolean(pecas, OpType.Add)

    # furos ao longo de toda a barra, simétricos a partir do centro
    n = int((meia_corda - t - PONTE_FURO_D) // PONTE_PASSO)
    furo = Manifold.cylinder(PONTE_ALT + 2, PONTE_FURO_D / 2, PONTE_FURO_D / 2, 32)
    furos_ = [furo.translate((afastamento, k * PONTE_PASSO, z_sup - PONTE_ALT - 1))
              for k in range(-n, n + 1)]
    peca = peca - Manifold.batch_boolean(furos_, OpType.Add)

    if not posicao_impressao:
        return peca.translate((0, 0, PE_ALTURA)), 2 * n + 1  # montada no vaso
    peca = peca.rotate((180, 0, 0))            # de cabeça para baixo
    (x0, y0, zmin, x1, y1, _) = peca.bounding_box()
    return peca.translate((-(x0 + x1) / 2, -(y0 + y1) / 2, -zmin)), 2 * n + 1


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


def exportar(peca, saida):
    m = peca.to_mesh()
    malha = trimesh.Trimesh(vertices=m.vert_properties[:, :3], faces=m.tri_verts)
    malha.export(saida)
    print(f"{saida}: {len(malha.faces)} triângulos, "
          f"fechada={malha.is_watertight}, volume={malha.volume / 1000:.1f} cm³, "
          f"dimensões={np.round(malha.extents, 1).tolist()} mm")


def main():
    exportar(gerar(), "vaso_orquidea_rugas_pes.stl")
    for nome, afastamento in PONTES:
        peca, n_furos = ponte(afastamento)
        exportar(peca, f"ponte_arames_{nome}.stl")
        print(f"   ponte {nome}: {n_furos} furos")


if __name__ == "__main__":
    main()
