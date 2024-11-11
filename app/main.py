from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import torch
import chess
import json
import asyncio
from typing import Dict
import numpy as np
import torch.nn.functional as F
from app.ChessNetwork import ChessNetwork
from app.ChessEnvironment import ChessEnvironment
from app.utils import move_to_index, index_to_move

class ChessGame:
    def __init__(self, model_path='app/models/chess_model_iter_9.pt'):
        # Inicializar el modelo
        self.model = ChessNetwork(num_res_blocks=8, num_channels=128)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Inicializar el tablero y el entorno
        self.board = chess.Board()
        self.env = ChessEnvironment()
        
    async def get_ai_move(self) -> str:
        """Obtiene el movimiento de la AI"""
        self.env.board = self.board.copy()
        state = self.env._get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, _ = self.model(state_tensor)
            policy = F.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        
        legal_moves = list(self.board.legal_moves)
        legal_move_probs = np.array([policy[move_to_index(move, self.board)] 
                                   for move in legal_moves])
        legal_move_probs = legal_move_probs / np.sum(legal_move_probs)
        
        move_idx = np.argmax(legal_move_probs)
        move = legal_moves[move_idx]
        
        return move.uci()
    
    def make_move(self, move_uci: str) -> bool:
        """Realiza un movimiento en el tablero"""
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except ValueError:
            return False
    
    def get_game_state(self) -> Dict:
        """Obtiene el estado actual del juego"""
        return {
            'fen': self.board.fen(),
            'legal_moves': [move.uci() for move in self.board.legal_moves],
            'is_game_over': self.board.is_game_over(),
            'is_check': self.board.is_check(),
            'result': self.get_game_result()
        }
    
    def get_game_result(self) -> str:
        """Obtiene el resultado del juego"""
        if not self.board.is_game_over():
            return 'ongoing'
        
        if self.board.is_checkmate():
            return 'checkmate'
        elif self.board.is_stalemate():
            return 'stalemate'
        elif self.board.is_insufficient_material():
            return 'insufficient_material'
        elif self.board.is_fifty_moves():
            return 'fifty_moves'
        elif self.board.is_repetition():
            return 'repetition'
        return 'draw'

class ChessServer:
    def __init__(self):
        self.app = FastAPI()
        self.active_games: Dict[str, ChessGame] = {}
        
        # Configurar CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Configurar rutas WebSocket
        self.app.websocket("/ws/{game_id}")(self.websocket_endpoint)
    
    async def websocket_endpoint(self, websocket: WebSocket, game_id: str):
        await websocket.accept()
        
        try:
            # Crear nuevo juego si no existe
            if game_id not in self.active_games:
                self.active_games[game_id] = ChessGame()
            
            game = self.active_games[game_id]
            
            # Enviar estado inicial
            await websocket.send_json(game.get_game_state())
            
            # Bucle principal del juego
            while True:
                data = await websocket.receive_json()
                command = data.get('command')
                
                if command == 'move':
                    # Procesar movimiento del jugador
                    move = data.get('move')
                    if game.make_move(move):
                        # Enviar estado actualizado
                        await websocket.send_json(game.get_game_state())
                        
                        # Si el juego no ha terminado, obtener movimiento de la AI
                        if not game.board.is_game_over():
                            ai_move = await game.get_ai_move()
                            game.make_move(ai_move)
                            await websocket.send_json({
                                **game.get_game_state(),
                                'ai_move': ai_move
                            })
                    else:
                        await websocket.send_json({
                            'error': 'Invalid move'
                        })
                
                elif command == 'restart':
                    # Reiniciar juego
                    self.active_games[game_id] = ChessGame()
                    game = self.active_games[game_id]
                    await websocket.send_json(game.get_game_state())
                
        except Exception as e:
            await websocket.send_json({
                'error': str(e)
            })
        
        finally:
            # Limpiar el juego cuando se cierra la conexión
            if game_id in self.active_games:
                del self.active_games[game_id]

# Crear y ejecutar el servidor
chess_server = ChessServer()
app = chess_server.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)