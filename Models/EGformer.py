"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter, segment_coo
import torch.nn.utils.rnn as rnn_utils
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_max_neighbors_mask,
    get_pbc_distances,
    radius_graph_pbc,
    scatter_det
    
)
from ocpmodels.datasets import LmdbDataset
from ocpmodels.modules.scaling.compat import load_scales_compat

from ocpmodels.models.gemnet_oc.initializers import get_initializer
from ocpmodels.models.gemnet_oc.interaction_indices import (
    get_mixed_triplets,
    get_quadruplets,
    get_triplets,
)
from ocpmodels.models.gemnet_oc.layers.atom_update_block import OutputBlock
from ocpmodels.models.gemnet_oc.layers.base_layers import Dense, ResidualLayer
from ocpmodels.models.gemnet_oc.layers.efficient import BasisEmbedding
from ocpmodels.models.gemnet_oc.layers.embedding_block import AtomEmbedding, EdgeEmbedding
from ocpmodels.models.gemnet_oc.layers.force_scaler import ForceScaler
from ocpmodels.models.gemnet_oc.layers.interaction_block import InteractionBlock
from ocpmodels.models.gemnet_oc.layers.radial_basis import RadialBasis
from ocpmodels.models.gemnet_oc.layers.spherical_basis import CircularBasisLayer, SphericalBasisLayer
from ocpmodels.models.gemnet_oc.utils import (
    get_angle,
    get_edge_id,
    get_inner_idx,
    inner_product_clamped,
    mask_neighbors,
    repeat_blocks,
)
from ocpmodels.trainers import ForcesTrainer
from ocpmodels import models
from ocpmodels.models.gemnet_oc.gemnet_oc import GemNetOC
from Models.encoder_layer import EncoderLayer

@registry.register_model('EGformer')
class EGformer(GemNetOC):
    """
    Arguments
    ---------
    num_atoms (int): Unused argument
    bond_feat_dim (int): Unused argument
    num_targets: int
        Number of prediction targets.

    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    num_blocks: int
        Number of building blocks to be stacked.

    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_trip_in: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        before the bilinear layer.
    emb_size_trip_out: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        after the bilinear layer.
    emb_size_quad_in: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        before the bilinear layer.
    emb_size_quad_out: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        after the bilinear layer.
    emb_size_aint_in: int
        Embedding size in the atom interaction before the bilinear layer.
    emb_size_aint_out: int
        Embedding size in the atom interaction after the bilinear layer.
    emb_size_rbf: int
        Embedding size of the radial basis transformation.
    emb_size_cbf: int
        Embedding size of the circular basis transformation (one angle).
    emb_size_sbf: int
        Embedding size of the spherical basis transformation (two angles).

    num_before_skip: int
        Number of residual blocks before the first skip connection.
    num_after_skip: int
        Number of residual blocks after the first skip connection.
    num_concat: int
        Number of residual blocks after the concatenation.
    num_atom: int
        Number of residual blocks in the atom embedding blocks.
    num_output_afteratom: int
        Number of residual blocks in the output blocks
        after adding the atom embedding.
    num_atom_emb_layers: int
        Number of residual blocks for transforming atom embeddings.
    num_global_out_layers: int
        Number of final residual blocks before the output.

    regress_forces: bool
        Whether to predict forces. Default: True
    direct_forces: bool
        If True predict forces based on aggregation of interatomic directions.
        If False predict forces based on negative gradient of energy potential.
    use_pbc: bool
        Whether to use periodic boundary conditions.
    scale_backprop_forces: bool
        Whether to scale up the energy and then scales down the forces
        to prevent NaNs and infs in backpropagated forces.

    cutoff: float
        Embedding cutoff for interatomic connections and embeddings in Angstrom.
    cutoff_qint: float
        Quadruplet interaction cutoff in Angstrom.
        Optional. Uses cutoff per default.
    cutoff_aeaint: float
        Edge-to-atom and atom-to-edge interaction cutoff in Angstrom.
        Optional. Uses cutoff per default.
    cutoff_aint: float
        Atom-to-atom interaction cutoff in Angstrom.
        Optional. Uses maximum of all other cutoffs per default.
    max_neighbors: int
        Maximum number of neighbors for interatomic connections and embeddings.
    max_neighbors_qint: int
        Maximum number of quadruplet interactions per embedding.
        Optional. Uses max_neighbors per default.
    max_neighbors_aeaint: int
        Maximum number of edge-to-atom and atom-to-edge interactions per embedding.
        Optional. Uses max_neighbors per default.
    max_neighbors_aint: int
        Maximum number of atom-to-atom interactions per atom.
        Optional. Uses maximum of all other neighbors per default.
    enforce_max_neighbors_strictly: bool
        When subselected edges based on max_neighbors args, arbitrarily
        select amongst degenerate edges to have exactly the correct number.
    rbf: dict
        Name and hyperparameters of the radial basis function.
    rbf_spherical: dict
        Name and hyperparameters of the radial basis function used as part of the
        circular and spherical bases.
        Optional. Uses rbf per default.
    envelope: dict
        Name and hyperparameters of the envelope function.
    cbf: dict
        Name and hyperparameters of the circular basis function.
    sbf: dict
        Name and hyperparameters of the spherical basis function.
    extensive: bool
        Whether the output should be extensive (proportional to the number of atoms)
    forces_coupled: bool
        If True, enforce that |F_st| = |F_ts|. No effect if direct_forces is False.
    output_init: str
        Initialization method for the final dense layer.
    activation: str
        Name of the activation function.
    scale_file: str
        Path to the pytorch file containing the scaling factors.

    quad_interaction: bool
        Whether to use quadruplet interactions (with dihedral angles)
    atom_edge_interaction: bool
        Whether to use atom-to-edge interactions
    edge_atom_interaction: bool
        Whether to use edge-to-atom interactions
    atom_interaction: bool
        Whether to use atom-to-atom interactions

    scale_basis: bool
        Whether to use a scaling layer in the raw basis function for better
        numerical stability.
    qint_tags: list
        Which atom tags to use quadruplet interactions for.
        0=sub-surface bulk, 1=surface, 2=adsorbate atoms.
    latent: bool
        Decide if output the latent space or not.
    
    """
    def __init__(
        self,
        num_atoms: Optional[int],
        bond_feat_dim: int,
        num_targets: int,
        num_spherical=7,
        num_radial=128,
        num_blocks=4,
        emb_size_atom=256,
        emb_size_edge=512,
        emb_size_trip_in=64,
        emb_size_trip_out=64,
        emb_size_quad_in=32,
        emb_size_quad_out=32,
        emb_size_aint_in=64,
        emb_size_aint_out=64,
        emb_size_rbf=16,
        emb_size_cbf=16,
        emb_size_sbf=32,
        num_before_skip=2,
        num_after_skip=2,
        num_concat=1,
        num_atom=3,
        num_output_afteratom=3,
        num_atom_emb_layers = 0,
        num_global_out_layers = 2,
        regress_forces = True,
        direct_forces = False,
        use_pbc = True,
        scale_backprop_forces = False,
        cutoff = 12.0,
        cutoff_qint = 12.0,
        cutoff_aeaint = 12.0,
        cutoff_aint = 12.0,
        max_neighbors = 30,
        max_neighbors_qint =8,
        max_neighbors_aeaint =20,
        max_neighbors_aint = 1000,
        enforce_max_neighbors_strictly = True,
        rbf = {"name": "gaussian"},
        rbf_spherical = None,
        envelope = {"name": "polynomial", "exponent": 5},
        cbf = {"name": "spherical_harmonics"},
        sbf = {"name": "spherical_harmonics"},
        extensive = True,
        forces_coupled = False,
        output_init = "HeOrthogonal",
        activation = "silu",
        quad_interaction = True,
        atom_edge_interaction = True,
        edge_atom_interaction = True,
        atom_interaction = True,
        scale_basis = False,
        qint_tags = [1, 2],
        num_elements = 83,
        otf_graph = True,
        scale_file = None,
        latent: bool = True,
        num_heads=1,
        emb_size_in=256,
        emb_size_trans=64,
        out_layer1=32,
        out_layer2=1,
        batch_size=2,
        **kwargs,  # backwards compatibility with deprecated arguments
    ):
        super().__init__(
                        num_atoms,
                        bond_feat_dim,
                        num_targets,
                        num_spherical=7,
                        num_radial=128,
                        num_blocks=4,
                        emb_size_atom=256,#256
                        emb_size_edge=512,#512
                        emb_size_trip_in=64,
                        emb_size_trip_out=64,
                        emb_size_quad_in=32,
                        emb_size_quad_out=32,
                        emb_size_aint_in=64,
                        emb_size_aint_out=64,
                        emb_size_rbf=16,
                        emb_size_cbf=16,
                        emb_size_sbf=32,
                        num_before_skip=2,
                        num_after_skip=2,
                        num_concat=1,
                        num_atom=3,
                        num_output_afteratom=3,
                        num_atom_emb_layers = 0,
                        num_global_out_layers = 2,
                        regress_forces = True,
                        direct_forces = False,
                        use_pbc = True,
                        scale_backprop_forces = False,
                        cutoff = 12.0,
                        cutoff_qint = 12.0,
                        cutoff_aeaint = 12.0,
                        cutoff_aint = 12.0,
                        max_neighbors = 30,
                        max_neighbors_qint =8,
                        max_neighbors_aeaint =20,
                        max_neighbors_aint = 1000,
                        enforce_max_neighbors_strictly = True,
                        rbf = {"name": "gaussian"},
                        rbf_spherical = None,
                        envelope = {"name": "polynomial", "exponent": 5},
                        cbf = {"name": "spherical_harmonics"},
                        sbf = {"name": "spherical_harmonics"},
                        extensive = True,
                        forces_coupled = False,
                        output_init = "HeOrthogonal",
                        activation = "silu",
                        quad_interaction = True,
                        atom_edge_interaction = True,
                        edge_atom_interaction = True,
                        atom_interaction = True,
                        scale_basis = False,
                        qint_tags = [1, 2],
                        num_elements = 83,
                        otf_graph = True,
                        scale_file = None,
                        num_heads=1,
                        emb_size_in=256,
                        emb_size_trans=64,
                        out_layer1=32,
                        out_layer2=1,
                        batch_size=2                        
                        )
        self.batch_size=batch_size
        self.num_heads=num_heads        
        self.out_layer1=out_layer1
        self.out_layer2=out_layer2
        self.lin_1=nn.Linear(emb_size_in,emb_size_trans)  
        self.encoder=EncoderLayer(emb_size_trans,4,emb_size_trans)
        self.layer_norm = nn.LayerNorm(emb_size_trans)        
        self.dense=nn.Sequential(nn.Linear(emb_size_trans,out_layer1),
                            nn.SiLU(),
                            nn.Linear(out_layer1,out_layer2)                                 
                            )

    def forward(self, data):
        pos = data.pos
        batch = data.batch
        
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.shape[0]
        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        ) = self.get_graphs_and_indices(data)
        # print('checkpoint1')
        _, idx_t = main_graph["edge_index"]

        (
            basis_rad_raw,
            basis_atom_update,
            basis_output,
            bases_qint,
            bases_e2e,
            bases_a2e,
            bases_e2a,
            basis_a2a_rad,
        ) = self.get_bases(
            main_graph=main_graph,
            a2a_graph=a2a_graph,
            a2ee2a_graph=a2ee2a_graph,
            qint_graph=qint_graph,
            trip_idx_e2e=trip_idx_e2e,
            trip_idx_a2e=trip_idx_a2e,
            trip_idx_e2a=trip_idx_e2a,
            quad_idx=quad_idx,
            num_atoms=num_atoms,
        )
        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, basis_rad_raw, main_graph["edge_index"])
        # (nEdges, emb_size_edge)

        x_E, _ = self.out_blocks[0](h, m, basis_output, idx_t)
        # print(x_E.shape)
        # xs_E, _ = [x_E], [x_F]
        xs_E = [x_E]
        # (nAtoms, num_targets), (nEdges, num_targets)

        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                bases_qint=bases_qint,
                bases_e2e=bases_e2e,
                bases_a2e=bases_a2e,
                bases_e2a=bases_e2a,
                basis_a2a_rad=basis_a2a_rad,
                basis_atom_update=basis_atom_update,
                edge_index_main=main_graph["edge_index"],
                a2ee2a_graph=a2ee2a_graph,
                a2a_graph=a2a_graph,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            x_E, _ = self.out_blocks[i + 1](h, m, basis_output, idx_t)
            # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
            xs_E.append(x_E)
            # xs_F.append(x_F)

        E_t = torch.stack(xs_E, dim=-1)
        # E_t= torch.mean(E_t)
        # print('checkpoint1')
        
        # num_atoms=[]
        if self.batch_size==1:
            E_t = torch.sum(E_t,dim=-1)
        else:
            num_atoms=[data[j].natoms[0] for j in range(self.batch_size)]
        # for j in range(self.batch_size):
        #         num_atoms.append(data[j].natoms[0])                
        # split_indices =(np.cumsum(num_atoms)[:-1])[0]
            E_t = torch.sum(E_t,dim=-1)
            E_t = torch.split(E_t,num_atoms,dim=0)
        # print('checkpoint2')
            E_t = rnn_utils.pad_sequence(E_t, batch_first=True, padding_value=0)
        E_t=self.lin_1(E_t)
        E_t = self.layer_norm(E_t)

        E_t,encoder_attention_weights=self.encoder(E_t,mask=None)
        # nMolecules = torch.max(batch) + 1
        E_t=torch.sum(E_t,dim=1)       
        E_t = self.layer_norm(E_t)   
        # print('checkpoint3')    
        E_t=torch.unsqueeze(E_t,dim=0)        
        E_t = E_t.permute(1, 0, 2)        
        # E_t = scatter(
        #         E_t, batch, dim=0, dim_size=nMolecules, reduce="add"
        #     )  
        E_t=self.dense(E_t)

        return E_t